//! Missing-value-aware, dtype-tolerant column extraction shared by core, MCP and CLI.
//!
//! Every estimator that reads user columns goes through this module (or the thin MCP
//! wrappers built on it), so that one set of rules decides what a column means:
//!
//! 1. **Missing means null or NaN**, as R's `is.na`. `±Inf` is a value, not a missing
//!    cell; numeric extraction refuses it with [`EconError::NonFiniteValues`] unless the
//!    request allows it ([`ColumnRequest::allow_infinite`]).
//! 2. **Numeric dtypes.** `Float64`, `Float32`, `Int8`..`Int128`, `UInt8`..`UInt128` and
//!    `Decimal` are cast to f64 (integers above 2^53 lose precision, as in R). `Boolean`
//!    is 0/1, as R's `as.numeric(logical)`, unless [`ColumnRequest::allow_bool`] turns it
//!    off. `String`, `Categorical`, `Enum`, temporal and nested columns are refused with
//!    [`EconError::NonNumericColumn`]: text is never cast to numbers.
//! 3. **Rows with a missing value** are dropped from every requested column
//!    ([`NaPolicy::Omit`], the default; R `na.omit` / `complete.cases`) or refused
//!    ([`NaPolicy::Fail`], R `na.fail`) with [`EconError::NullValues`] naming the first
//!    requested column that has one. [`MissingSummary`] reports what was dropped.
//! 4. **Factor levels** follow R `factor(x)` (`sort(unique(x))`): numbers by value
//!    (`2 < 10`, `-2 < -1`), text by byte order, booleans `FALSE < TRUE`, dates
//!    chronologically, `Categorical` by label and `Enum` in its declared order. Byte order
//!    is R's order under `LC_COLLATE=C`; R in a UTF-8 locale sorts text
//!    case-insensitively (`a A b B` instead of `A B a b`). An extracted factor keeps the
//!    levels observed in the kept rows (R `model.frame`, `drop.unused.levels = TRUE`);
//!    [`factor_options`] keeps every level observed in a non-missing cell.
//! 5. **A time index** holds integers, whole-number floats or dates (days since
//!    1970-01-01); a fractional value or a text column is an error naming the column.
//! 6. **Errors name the column**: `ColumnNotFound` only for a column that does not
//!    exist, and `NullValues`, `NonNumericColumn` or `NonFiniteValues` for one that exists.
//! 7. A column whose every cell is null is accepted in every role as all-missing,
//!    whatever its dtype: polars reads an all-empty CSV column as `String`, where R reads
//!    logical `NA`.
//!
//! Every pass over the rows is O(n x k). The row mask, the kept rows, the column values,
//! the time index values and the factor codes are allocated through [`crate::resource`], and
//! so is the n x k copy of [`Columns::matrix`] (a name repeated in `names` is copied once per
//! occurrence, so k can exceed the columns extracted). [`Columns::numeric_array`] copies one
//! extracted column infallibly; per-level buffers (counts, level remaps) are sized by the
//! number of levels.
//!
//! ```
//! use p2a_core::data::columns::{ColumnRequest, NaPolicy};
//! use polars::prelude::*;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let df = df! { "y" => [Some(1.5), None, Some(3.0)], "g" => [Some(10i64), Some(2), Some(2)] }?;
//! let cols = ColumnRequest::new(&df).numeric("y").factor("g").na(NaPolicy::Omit).extract()?;
//! assert_eq!(cols.n(), 2);
//! assert_eq!(cols.factor("g").levels, ["2", "10"]);          // numeric order, as R factor()
//! assert_eq!(cols.missing().note().unwrap(), "1 of 3 rows dropped because of missing values (y: 1)");
//! # Ok(())
//! # }
//! ```

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use ndarray::Array1;
use polars::prelude::{
    ChunkFullNull, Column, DataFrame, DataType, Float64Chunked, PolarsError, StringChunked,
};
use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignError, DesignMatrix, get_column_names};
use crate::resource::{try_vec, try_with_capacity, try_zeros2};

/// What to do with rows that have a missing value (null, or NaN in a float column) in any
/// requested column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NaPolicy {
    /// Drop the row from every requested column (R `na.omit` / `complete.cases`). Default.
    #[default]
    Omit,
    /// Refuse with `EconError::NullValues { column, count }` for the first requested column
    /// (request order) that has a missing value (R `na.fail`).
    Fail,
}

/// Rows removed because of missing values. `n_dropped == 0` means nothing was removed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingSummary {
    /// Rows in the input.
    pub n_total: usize,
    /// Rows used.
    pub n_used: usize,
    /// Rows dropped (`n_total - n_used`).
    pub n_dropped: usize,
    /// Missing cells per requested column, request order, only columns with at least one.
    /// A row missing in two columns counts in both. Serialized as an object
    /// {column: count}; a streaming serializer keeps request order, `serde_json::Value`
    /// sorts the keys.
    #[serde(with = "by_column_map")]
    pub by_column: Vec<(String, usize)>,
}

impl MissingSummary {
    /// Nothing dropped from `n` rows.
    pub fn none(n: usize) -> Self {
        MissingSummary {
            n_total: n,
            n_used: n,
            n_dropped: 0,
            by_column: Vec::new(),
        }
    }

    /// `n_dropped == 0` (used by `#[serde(skip_serializing_if = "MissingSummary::is_empty")]`).
    pub fn is_empty(&self) -> bool {
        self.n_dropped == 0
    }

    /// "3 of 120 rows dropped because of missing values (y: 2, x1: 1)"; `None` when empty.
    pub fn note(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut note = format!(
            "{} of {} rows dropped because of missing values",
            self.n_dropped, self.n_total
        );
        if !self.by_column.is_empty() {
            let list: Vec<String> = self
                .by_column
                .iter()
                .map(|(column, count)| format!("{column}: {count}"))
                .collect();
            note.push_str(&format!(" ({})", list.join(", ")));
        }
        Some(note)
    }

    /// Sum two summaries of independent samples (two-sample tests that drop per column).
    /// The per-column counts of `a` come first; a column present in both is summed.
    pub fn merge_samples(a: &MissingSummary, b: &MissingSummary) -> MissingSummary {
        let mut by_column = a.by_column.clone();
        for (name, count) in &b.by_column {
            match by_column.iter_mut().find(|(n, _)| n == name) {
                Some((_, c)) => *c = c.saturating_add(*count),
                None => by_column.push((name.clone(), *count)),
            }
        }
        MissingSummary {
            n_total: a.n_total.saturating_add(b.n_total),
            n_used: a.n_used.saturating_add(b.n_used),
            n_dropped: a.n_dropped.saturating_add(b.n_dropped),
            by_column,
        }
    }

    /// The note of a summary that [`merge_samples`](Self::merge_samples) built from two
    /// columns of one frame: "30 of 44 values dropped because of missing values (two samples
    /// of 22; x: 15, y: 15)"; `None` when empty. Each sample drops its own missing values, so
    /// the counts are of values, not rows; a column used as both samples is counted in each.
    /// [`note`](Self::note) is unchanged, and the serialized summary keeps the merged totals.
    pub fn two_sample_note(&self) -> Option<String> {
        if self.is_empty() {
            return None;
        }
        let mut detail: Vec<String> = Vec::new();
        if self.n_total.is_multiple_of(2) {
            detail.push(format!("two samples of {}", self.n_total / 2));
        }
        if !self.by_column.is_empty() {
            let list: Vec<String> = self
                .by_column
                .iter()
                .map(|(column, count)| format!("{column}: {count}"))
                .collect();
            detail.push(list.join(", "));
        }
        let mut note = format!(
            "{} of {} values dropped because of missing values",
            self.n_dropped, self.n_total
        );
        if !detail.is_empty() {
            note.push_str(&format!(" ({})", detail.join("; ")));
        }
        Some(note)
    }
}

impl fmt::Display for MissingSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.note() {
            Some(note) => f.write_str(&note),
            None => Ok(()),
        }
    }
}

/// `MissingSummary::by_column` as a map from column name to count, in request order.
mod by_column_map {
    use std::fmt;

    use serde::de::{MapAccess, Visitor};
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        pairs: &[(String, usize)],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_map(pairs.iter().map(|(column, count)| (column, count)))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<(String, usize)>, D::Error> {
        struct Pairs;
        impl<'de> Visitor<'de> for Pairs {
            type Value = Vec<(String, usize)>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a map from column name to missing-cell count")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut pairs = Vec::new();
                while let Some(entry) = map.next_entry::<String, usize>()? {
                    pairs.push(entry);
                }
                Ok(pairs)
            }
        }
        deserializer.deserialize_map(Pairs)
    }
}

/// The role a requested column plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Numeric,
    Factor,
    Time,
}

/// A request for row-aligned columns. Duplicate names are extracted once.
#[derive(Debug, Clone)]
pub struct ColumnRequest<'a> {
    df: &'a DataFrame,
    entries: Vec<(String, Role)>,
    policy: NaPolicy,
    allow_bool: bool,
    allow_infinite: bool,
    min_rows: (usize, String),
}

impl<'a> ColumnRequest<'a> {
    /// Policy `Omit`, booleans accepted as 0/1, infinite values refused, `min_rows` 1.
    pub fn new(df: &'a DataFrame) -> Self {
        ColumnRequest {
            df,
            entries: Vec::new(),
            policy: NaPolicy::Omit,
            allow_bool: true,
            allow_infinite: false,
            min_rows: (1, "column extraction".to_string()),
        }
    }

    fn push(mut self, name: &str, role: Role) -> Self {
        self.entries.push((name.to_string(), role));
        self
    }

    /// A numeric column (rule 2 of the module docs).
    pub fn numeric(self, name: &str) -> Self {
        self.push(name, Role::Numeric)
    }

    /// Several numeric columns, in order.
    pub fn numerics(self, names: &[&str]) -> Self {
        names.iter().fold(self, |r, name| r.numeric(name))
    }

    /// A categorical column (rule 4 of the module docs).
    pub fn factor(self, name: &str) -> Self {
        self.push(name, Role::Factor)
    }

    /// Several categorical columns, in order.
    pub fn factors(self, names: &[&str]) -> Self {
        names.iter().fold(self, |r, name| r.factor(name))
    }

    /// Whole-number time index: integers, integral floats, Date (days since 1970-01-01).
    pub fn time_index(self, name: &str) -> Self {
        self.push(name, Role::Time)
    }

    /// The missing-value policy (default [`NaPolicy::Omit`]).
    pub fn na(mut self, policy: NaPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Accept `Boolean` columns as 0/1 in a numeric role (default true).
    pub fn allow_bool(mut self, yes: bool) -> Self {
        self.allow_bool = yes;
        self
    }

    /// Keep `±Inf` in numeric columns instead of refusing them (default false).
    pub fn allow_infinite(mut self, yes: bool) -> Self {
        self.allow_infinite = yes;
        self
    }

    /// After deletion fewer than `n` rows -> `EconError::InsufficientData` naming `context`.
    pub fn min_rows(mut self, n: usize, context: &str) -> Self {
        self.min_rows = (n, context.to_string());
        self
    }

    /// Extract the requested columns.
    ///
    /// The checks run in this order: every entry in request order exists
    /// (`ColumnNotFound`) and passes its role's dtype rule (`NonNumericColumn` for a
    /// numeric entry, `InvalidSpecification` for a factor or time index); every time
    /// index holds whole numbers; with [`NaPolicy::Fail`], the first entry with a missing
    /// cell gives `NullValues`; unless infinite values are allowed, the first numeric
    /// entry with `±Inf` in a kept row gives `NonFiniteValues`; fewer kept rows than
    /// `min_rows` gives `InsufficientData`.
    pub fn extract(self) -> EconResult<Columns> {
        let df = self.df;
        let n = df.height();

        let mut entries: Vec<(String, Role)> = Vec::with_capacity(self.entries.len());
        for entry in self.entries {
            if !entries.contains(&entry) {
                entries.push(entry);
            }
        }

        // 1. Existence and the role's dtype rule, entry by entry; 2. whole-number time
        // indexes (the first such error is reported after every dtype check).
        let mut raws: Vec<Raw> = Vec::with_capacity(entries.len());
        let mut deferred: Option<EconError> = None;
        for (name, role) in &entries {
            let col = column(df, name)?;
            let raw = match role {
                Role::Numeric => {
                    Raw::Numeric(float64_strict(col, self.allow_bool).ok_or_else(|| {
                        EconError::NonNumericColumn {
                            column: name.clone(),
                        }
                    })?)
                }
                Role::Factor => Raw::Factor(raw_factor(col, name)?),
                Role::Time => {
                    check_time_dtype(col, name)?;
                    match time_values(col, name) {
                        Ok(values) => Raw::Time(values),
                        Err(e) => {
                            deferred.get_or_insert(e);
                            Raw::Time(Vec::new())
                        }
                    }
                }
            };
            raws.push(raw);
        }
        if let Some(e) = deferred {
            return Err(e);
        }

        let missing_counts: Vec<usize> = raws.iter().map(Raw::missing_count).collect();

        // 3. NaPolicy::Fail: the first entry, in request order, with a missing cell.
        if self.policy == NaPolicy::Fail
            && let Some(((name, _), &count)) = entries
                .iter()
                .zip(&missing_counts)
                .find(|(_, count)| **count > 0)
        {
            return Err(EconError::NullValues {
                column: name.clone(),
                count,
            });
        }

        // 4. The row mask (true = some requested column is missing) and the kept rows.
        let mut dropped = try_vec(n, false, "row mask")?;
        for raw in &raws {
            raw.mark_missing(&mut dropped);
        }
        let mut rows: Vec<usize> = try_with_capacity(n, "kept rows")?;
        rows.extend(
            dropped
                .iter()
                .enumerate()
                .filter(|(_, d)| !**d)
                .map(|(i, _)| i),
        );
        let n_used = rows.len();

        // 5. Infinite values in kept rows of numeric entries.
        if !self.allow_infinite {
            for ((name, _), raw) in entries.iter().zip(&raws) {
                if let Raw::Numeric(ca) = raw {
                    let count = ca
                        .iter()
                        .zip(&dropped)
                        .filter(|(v, d)| !**d && v.is_some_and(f64::is_infinite))
                        .count();
                    if count > 0 {
                        return Err(EconError::NonFiniteValues {
                            column: name.clone(),
                            count,
                        });
                    }
                }
            }
        }

        // 6. Enough rows left.
        let (required, context) = &self.min_rows;
        if n_used < *required {
            return Err(EconError::InsufficientData {
                required: *required,
                provided: n_used,
                context: format!("{context}: {n_used} of {n} rows have no missing value"),
            });
        }

        // The missing cells per requested name, first-request order.
        let mut by_column: Vec<(String, usize)> = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        for ((name, _), &count) in entries.iter().zip(&missing_counts) {
            if seen.contains(&name.as_str()) {
                continue;
            }
            seen.push(name);
            if count > 0 {
                by_column.push((name.clone(), count));
            }
        }

        // 7. Gather the kept values.
        let mut numerics = HashMap::new();
        let mut factors = HashMap::new();
        let mut times = HashMap::new();
        for ((name, _), raw) in entries.iter().zip(&raws) {
            match raw {
                Raw::Numeric(ca) => {
                    let mut values = try_with_capacity(n_used, "column values")?;
                    for (v, d) in ca.iter().zip(&dropped) {
                        if !*d && let Some(v) = v {
                            values.push(v);
                        }
                    }
                    numerics.insert(name.clone(), values);
                }
                Raw::Factor(raw) => {
                    factors.insert(name.clone(), raw.kept(&dropped, n_used)?);
                }
                Raw::Time(all) => {
                    let mut values = try_with_capacity(n_used, "time index values")?;
                    for (v, d) in all.iter().zip(&dropped) {
                        if !*d && let Some(v) = v {
                            values.push(*v);
                        }
                    }
                    times.insert(name.clone(), values);
                }
            }
        }

        Ok(Columns {
            rows,
            missing: MissingSummary {
                n_total: n,
                n_used,
                n_dropped: n - n_used,
                by_column,
            },
            numerics,
            factors,
            times,
        })
    }
}

/// One requested column over all rows, after its dtype rule.
enum Raw {
    /// Null or NaN is missing.
    Numeric(Float64Chunked),
    Factor(RawFactor),
    /// `None` is missing.
    Time(Vec<Option<i64>>),
}

impl Raw {
    fn missing_count(&self) -> usize {
        match self {
            Raw::Numeric(ca) => ca.iter().filter(|v| v.is_none_or(f64::is_nan)).count(),
            Raw::Factor(f) => f.codes.iter().filter(|c| c.is_none()).count(),
            Raw::Time(t) => t.iter().filter(|v| v.is_none()).count(),
        }
    }

    fn mark_missing(&self, dropped: &mut [bool]) {
        match self {
            Raw::Numeric(ca) => {
                for (d, v) in dropped.iter_mut().zip(ca.iter()) {
                    *d |= v.is_none_or(f64::is_nan);
                }
            }
            Raw::Factor(f) => {
                for (d, c) in dropped.iter_mut().zip(&f.codes) {
                    *d |= c.is_none();
                }
            }
            Raw::Time(t) => {
                for (d, v) in dropped.iter_mut().zip(t) {
                    *d |= v.is_none();
                }
            }
        }
    }
}

/// A factor over all rows: levels observed in any non-missing cell.
struct RawFactor {
    codes: Vec<Option<u32>>,
    levels: Vec<String>,
    numeric_levels: Vec<f64>,
}

impl RawFactor {
    /// The factor on the kept rows, with the levels observed there (in the same order).
    fn kept(&self, dropped: &[bool], n_used: usize) -> EconResult<Factor> {
        let n_levels = self.levels.len();
        let mut used = try_vec(n_levels, false, "factor levels")?;
        for (c, d) in self.codes.iter().zip(dropped) {
            if !*d && let Some(c) = c {
                used[*c as usize] = true;
            }
        }
        let mut new_code = try_vec(n_levels, 0u32, "factor levels")?;
        let mut levels = Vec::new();
        let mut numeric_levels = Vec::new();
        for (old, _) in used.iter().enumerate().filter(|(_, u)| **u) {
            new_code[old] = u32::try_from(levels.len()).map_err(|_| too_many_levels())?;
            levels.push(self.levels[old].clone());
            if let Some(&v) = self.numeric_levels.get(old) {
                numeric_levels.push(v);
            }
        }
        let mut codes = try_with_capacity(n_used, "factor codes")?;
        for (c, d) in self.codes.iter().zip(dropped) {
            if !*d && let Some(c) = c {
                codes.push(new_code[*c as usize]);
            }
        }
        Ok(Factor {
            levels,
            codes,
            numeric_levels,
        })
    }
}

/// Extracted columns; every vector has `n()` entries, aligned on `rows`.
#[derive(Debug, Clone)]
pub struct Columns {
    rows: Vec<usize>,
    missing: MissingSummary,
    numerics: HashMap<String, Vec<f64>>,
    factors: HashMap<String, Factor>,
    times: HashMap<String, Vec<i64>>,
}

impl Columns {
    /// Number of kept rows.
    pub fn n(&self) -> usize {
        self.rows.len()
    }

    /// Original row index (0-based, ascending) of each kept row.
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// What was dropped.
    pub fn missing(&self) -> &MissingSummary {
        &self.missing
    }

    /// The kept values of a numeric column.
    ///
    /// # Panics
    /// If `name` was not requested as numeric (a programming error, never user input).
    pub fn numeric(&self, name: &str) -> &[f64] {
        match self.numerics.get(name) {
            Some(values) => values,
            None => panic!("column '{name}' was not requested as numeric"),
        }
    }

    /// The kept values of a numeric column as an array.
    ///
    /// # Panics
    /// If `name` was not requested as numeric.
    pub fn numeric_array(&self, name: &str) -> Array1<f64> {
        Array1::from_vec(self.numeric(name).to_vec())
    }

    /// Design matrix over numeric columns in `names` order; "(Intercept)" first when
    /// `intercept`. The n x (`names.len()` + intercept) matrix is allocated through
    /// [`crate::resource`]: a matrix above the allocation ceiling is
    /// [`EconError::ResourceLimit`].
    ///
    /// # Panics
    /// If a name was not requested as numeric.
    pub fn matrix(&self, names: &[&str], intercept: bool) -> EconResult<DesignMatrix> {
        let n = self.n();
        let k = names.len().saturating_add(usize::from(intercept));
        let mut data = try_zeros2(n, k, "design matrix")?;
        let mut column_names = try_with_capacity(k, "design matrix column names")?;
        let mut j = 0;
        if intercept {
            data.column_mut(0).fill(1.0);
            column_names.push("(Intercept)".to_string());
            j = 1;
        }
        for &name in names {
            for (cell, &v) in data.column_mut(j).iter_mut().zip(self.numeric(name)) {
                *cell = v;
            }
            column_names.push(name.to_string());
            j += 1;
        }
        Ok(DesignMatrix {
            data,
            column_names,
            has_intercept: intercept,
            n_obs: n,
            n_features: k,
        })
    }

    /// A categorical column on the kept rows.
    ///
    /// # Panics
    /// If `name` was not requested as a factor.
    pub fn factor(&self, name: &str) -> &Factor {
        match self.factors.get(name) {
            Some(factor) => factor,
            None => panic!("column '{name}' was not requested as a factor"),
        }
    }

    /// A time index on the kept rows.
    ///
    /// # Panics
    /// If `name` was not requested as a time index.
    pub fn time_index(&self, name: &str) -> &[i64] {
        match self.times.get(name) {
            Some(values) => values,
            None => panic!("column '{name}' was not requested as a time index"),
        }
    }
}

/// A categorical column: labels in R `factor()` order and one code per kept row.
#[derive(Debug, Clone, PartialEq)]
pub struct Factor {
    /// Level labels: strings as stored (no quotes), integers in decimal, floats in Rust's
    /// shortest round-trip form ("2", "1.5", "Inf"), booleans "FALSE"/"TRUE", dates
    /// "YYYY-MM-DD".
    pub levels: Vec<String>,
    /// 0-based level index of each kept row.
    pub codes: Vec<u32>,
    /// Numeric value of each level for numeric source columns, else empty.
    pub numeric_levels: Vec<f64>,
}

impl Factor {
    /// Number of levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Label of kept row `i`.
    ///
    /// # Panics
    /// If `i >= codes.len()`.
    pub fn label(&self, i: usize) -> &str {
        &self.levels[self.codes[i] as usize]
    }

    /// Number of rows per level.
    pub fn counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.levels.len()];
        for &c in &self.codes {
            counts[c as usize] += 1;
        }
        counts
    }

    /// Re-order levels to `order` (every observed label must appear in `order`; labels in
    /// `order` that are never observed keep a count of 0). The error names the first
    /// unknown label with the order's length and at most its first five labels, or a label
    /// `order` repeats. For a numeric factor a label that is never
    /// observed gets its parsed value as its numeric level; one that does not parse as a
    /// number (or parses as NaN) is refused by name.
    pub fn with_level_order(&self, order: &[String]) -> EconResult<Factor> {
        let mut position: HashMap<&str, u32> = HashMap::with_capacity(order.len());
        for (i, label) in order.iter().enumerate() {
            let i = u32::try_from(i).map_err(|_| too_many_levels())?;
            if position.insert(label.as_str(), i).is_some() {
                return Err(EconError::InvalidSpecification {
                    message: format!("level '{}' appears twice in the level order", shown(label)),
                });
            }
        }
        let mut remap = Vec::with_capacity(self.levels.len());
        for label in &self.levels {
            match position.get(label.as_str()) {
                Some(&p) => remap.push(p),
                None => {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "level '{}' is not in the given level order ({})",
                            shown(label),
                            order_preview(order)
                        ),
                    });
                }
            }
        }
        let codes = self.codes.iter().map(|&c| remap[c as usize]).collect();
        let numeric_levels = if self.numeric_levels.is_empty() {
            Vec::new()
        } else {
            let old: HashMap<&str, f64> = self
                .levels
                .iter()
                .map(String::as_str)
                .zip(self.numeric_levels.iter().copied())
                .collect();
            let mut values = Vec::with_capacity(order.len());
            for label in order {
                let value = match old.get(label.as_str()) {
                    Some(&v) => v,
                    None => match label.parse::<f64>() {
                        Ok(v) if !v.is_nan() => v,
                        _ => {
                            return Err(EconError::InvalidSpecification {
                                message: format!(
                                    "level '{}' in the level order is not a number, \
                                     but the column levels are numeric",
                                    shown(label)
                                ),
                            });
                        }
                    },
                };
                values.push(value);
            }
            values
        };
        Ok(Factor {
            levels: order.to_vec(),
            codes,
            numeric_levels,
        })
    }
}

/// A label as a refusal shows it: at most [`SHOWN_LABEL_BYTES`] bytes (cut at a character
/// boundary), then "...".
fn shown(label: &str) -> String {
    if label.len() <= SHOWN_LABEL_BYTES {
        return label.to_string();
    }
    let cut = (0..=SHOWN_LABEL_BYTES)
        .rev()
        .find(|&i| label.is_char_boundary(i))
        .unwrap_or(0);
    format!("{}...", &label[..cut])
}

/// Bytes of a label a refusal shows.
const SHOWN_LABEL_BYTES: usize = 24;

/// "3 labels: 'a', 'b', 'c'" for a refusal: the length and at most the first five labels,
/// each cut by [`shown`], so the message stays bounded however long the caller's order and
/// its labels are.
fn order_preview(order: &[String]) -> String {
    const SHOWN: usize = 5;
    let mut labels: Vec<String> = order
        .iter()
        .take(SHOWN)
        .map(|label| format!("'{}'", shown(label)))
        .collect();
    if order.len() > SHOWN {
        labels.push("...".to_string());
    }
    let noun = if order.len() == 1 { "label" } else { "labels" };
    if labels.is_empty() {
        format!("{} {noun}", order.len())
    } else {
        format!("{} {noun}: {}", order.len(), labels.join(", "))
    }
}

/// One numeric column with a policy: the kept values and the summary.
pub fn numeric_column(
    df: &DataFrame,
    name: &str,
    policy: NaPolicy,
) -> EconResult<(Vec<f64>, MissingSummary)> {
    let mut cols = ColumnRequest::new(df).numeric(name).na(policy).extract()?;
    let values = cols.numerics.remove(name).unwrap_or_default();
    Ok((values, cols.missing))
}

/// Strict numeric check, missing (null or NaN) as `None`; for code that handles gaps itself
/// (Friedman blocks, time-series gap checks, LTMLE censoring). Booleans are 0/1; `±Inf`
/// is refused with `EconError::NonFiniteValues`.
pub fn numeric_options(df: &DataFrame, name: &str) -> EconResult<Vec<Option<f64>>> {
    let col = column(df, name)?;
    let ca = float64_strict(col, true).ok_or_else(|| EconError::NonNumericColumn {
        column: name.to_string(),
    })?;
    let mut out = try_with_capacity(ca.len(), "column values")?;
    let mut infinite = 0usize;
    for v in ca.iter() {
        match v {
            Some(x) if x.is_nan() => out.push(None),
            Some(x) => {
                if x.is_infinite() {
                    infinite += 1;
                }
                out.push(Some(x));
            }
            None => out.push(None),
        }
    }
    if infinite > 0 {
        return Err(EconError::NonFiniteValues {
            column: name.to_string(),
            count: infinite,
        });
    }
    Ok(out)
}

/// Factor codes with missing labels as `None`, plus the level labels (every label
/// observed in a non-missing cell, in R `factor()` order).
pub fn factor_options(df: &DataFrame, name: &str) -> EconResult<(Vec<Option<u32>>, Vec<String>)> {
    let raw = raw_factor(column(df, name)?, name)?;
    Ok((raw.codes, raw.levels))
}

/// Does `dtype` pass the numeric rule (integers, floats, Decimal; Boolean when
/// `allow_bool`)?
pub fn is_numeric_dtype(dtype: &DataType, allow_bool: bool) -> bool {
    dtype.is_numeric() || (allow_bool && dtype.is_bool())
}

/// The column, or `EconError::ColumnNotFound` with the available columns (typo suggestions).
pub fn column<'a>(df: &'a DataFrame, name: &str) -> EconResult<&'a Column> {
    df.column(name).map_err(|_| EconError::ColumnNotFound {
        column: name.to_string(),
        available: get_column_names(df),
    })
}

/// Map a `DesignError` to the matching `EconError`; never `ColumnNotFound` for an existing
/// column.
pub fn econ_error_from_design(err: DesignError, df: &DataFrame) -> EconError {
    match err {
        DesignError::ColumnNotFound(column) => EconError::ColumnNotFound {
            column,
            available: get_column_names(df),
        },
        DesignError::NonNumericColumn(column) => EconError::NonNumericColumn { column },
        DesignError::NullValues { column, count, .. } => EconError::NullValues { column, count },
        DesignError::NonFinite { column, count, .. } => {
            EconError::NonFiniteValues { column, count }
        }
        DesignError::EmptyDataset => EconError::EmptyDataset,
        DesignError::NoColumns => EconError::InvalidSpecification {
            message: "No independent variables specified".to_string(),
        },
        DesignError::ResourceLimit(message) => EconError::ResourceLimit(message),
        DesignError::PolarsError(e) => EconError::Internal(e.to_string()),
    }
}

/// The one strict numeric rule: the column as f64 when its dtype passes
/// [`is_numeric_dtype`], or a null column of the same length when every cell is null;
/// `None` otherwise. Never a non-strict cast that turns text into nulls.
pub(crate) fn float64_strict(column: &Column, allow_bool: bool) -> Option<Float64Chunked> {
    if column.null_count() == column.len() {
        return Some(Float64Chunked::full_null(
            column.name().clone(),
            column.len(),
        ));
    }
    if !is_numeric_dtype(column.dtype(), allow_bool) {
        return None;
    }
    let cast = column.strict_cast(&DataType::Float64).ok()?;
    cast.f64().ok().cloned()
}

fn too_many_levels() -> EconError {
    EconError::ResourceLimit("a factor has more levels than fit in 32-bit codes".to_string())
}

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

fn not_a_factor(name: &str, dtype: &DataType) -> EconError {
    invalid(format!(
        "Column '{name}' has type {dtype}, which cannot be used as a categorical variable."
    ))
}

fn polars_error(e: PolarsError) -> EconError {
    EconError::Internal(e.to_string())
}

/// Codes in first-seen order in one pass over the cells, then the unique keys sorted with
/// `cmp` and the codes renumbered to that order.
fn factor_from_keys<K: Copy + Eq + Hash>(
    values: impl Iterator<Item = Option<K>>,
    n: usize,
    cmp: impl Fn(&K, &K) -> Ordering,
) -> EconResult<(Vec<Option<u32>>, Vec<K>)> {
    let mut index: HashMap<K, u32> = HashMap::new();
    let mut keys: Vec<K> = Vec::new();
    let mut codes: Vec<Option<u32>> = try_with_capacity(n, "factor codes")?;
    for v in values {
        let code = match v {
            None => None,
            Some(key) => Some(match index.get(&key) {
                Some(&c) => c,
                None => {
                    let c = u32::try_from(keys.len()).map_err(|_| too_many_levels())?;
                    index.insert(key, c);
                    keys.push(key);
                    c
                }
            }),
        };
        codes.push(code);
    }
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| cmp(&keys[a], &keys[b]));
    let mut rank = vec![0u32; keys.len()];
    for (new, &old) in order.iter().enumerate() {
        rank[old] = u32::try_from(new).map_err(|_| too_many_levels())?;
    }
    for code in codes.iter_mut().flatten() {
        *code = rank[*code as usize];
    }
    Ok((codes, order.iter().map(|&i| keys[i]).collect()))
}

/// A text factor; `cmp` orders the labels.
fn string_factor<'s>(
    ca: &'s StringChunked,
    cmp: impl Fn(&&'s str, &&'s str) -> Ordering,
) -> EconResult<RawFactor> {
    let (codes, keys) = factor_from_keys(ca.iter(), ca.len(), cmp)?;
    Ok(RawFactor {
        codes,
        levels: keys.iter().map(|s| s.to_string()).collect(),
        numeric_levels: Vec::new(),
    })
}

/// A factor over all rows of `col`, with the levels observed in any non-missing cell.
fn raw_factor(col: &Column, name: &str) -> EconResult<RawFactor> {
    let n = col.len();
    if col.null_count() == n {
        return Ok(RawFactor {
            codes: try_vec(n, None, "factor codes")?,
            levels: Vec::new(),
            numeric_levels: Vec::new(),
        });
    }
    let dtype = col.dtype();
    if dtype.is_signed_integer() {
        let cast = col
            .strict_cast(&DataType::Int64)
            .map_err(|_| integers_out_of_range(name))?;
        let (codes, keys) =
            factor_from_keys(cast.i64().map_err(polars_error)?.iter(), n, Ord::cmp)?;
        return Ok(RawFactor {
            codes,
            levels: keys.iter().map(i64::to_string).collect(),
            numeric_levels: keys.iter().map(|&k| k as f64).collect(),
        });
    }
    if dtype.is_unsigned_integer() {
        let cast = col
            .strict_cast(&DataType::UInt64)
            .map_err(|_| integers_out_of_range(name))?;
        let (codes, keys) =
            factor_from_keys(cast.u64().map_err(polars_error)?.iter(), n, Ord::cmp)?;
        return Ok(RawFactor {
            codes,
            levels: keys.iter().map(u64::to_string).collect(),
            numeric_levels: keys.iter().map(|&k| k as f64).collect(),
        });
    }
    if dtype.is_float() || dtype.is_decimal() {
        let cast = col.strict_cast(&DataType::Float64).map_err(polars_error)?;
        // NaN is missing; -0.0 and 0.0 are one level (R: factor(c(-0, 0)) has one level "0").
        let bits = cast.f64().map_err(polars_error)?.iter().map(|v| {
            v.filter(|x| !x.is_nan())
                .map(|x| if x == 0.0 { 0.0f64 } else { x }.to_bits())
        });
        let (codes, keys) = factor_from_keys(bits, n, |a, b| {
            f64::from_bits(*a).total_cmp(&f64::from_bits(*b))
        })?;
        let values: Vec<f64> = keys.iter().map(|&b| f64::from_bits(b)).collect();
        return Ok(RawFactor {
            codes,
            levels: values.iter().map(|&v| float_label(v)).collect(),
            numeric_levels: values,
        });
    }
    match dtype {
        DataType::Boolean => {
            let (codes, keys) =
                factor_from_keys(col.bool().map_err(polars_error)?.iter(), n, Ord::cmp)?;
            Ok(RawFactor {
                codes,
                levels: keys
                    .iter()
                    .map(|&b| if b { "TRUE" } else { "FALSE" }.to_string())
                    .collect(),
                numeric_levels: Vec::new(),
            })
        }
        DataType::Date => {
            let cast = col.strict_cast(&DataType::Int32).map_err(polars_error)?;
            let (codes, keys) =
                factor_from_keys(cast.i32().map_err(polars_error)?.iter(), n, Ord::cmp)?;
            Ok(RawFactor {
                codes,
                levels: keys.iter().map(|&d| date_label(d)).collect(),
                numeric_levels: Vec::new(),
            })
        }
        DataType::String => string_factor(col.str().map_err(polars_error)?, Ord::cmp),
        DataType::Categorical(..) => {
            let labels = col.cast(&DataType::String).map_err(polars_error)?;
            string_factor(labels.str().map_err(polars_error)?, Ord::cmp)
        }
        DataType::Enum(categories, _) => {
            let declared: HashMap<&str, usize> = categories
                .categories()
                .values_iter()
                .enumerate()
                .map(|(i, label)| (label, i))
                .collect();
            let labels = col.cast(&DataType::String).map_err(polars_error)?;
            string_factor(labels.str().map_err(polars_error)?, |a, b| {
                declared.get(a).cmp(&declared.get(b))
            })
        }
        other => Err(not_a_factor(name, other)),
    }
}

fn integers_out_of_range(name: &str) -> EconError {
    invalid(format!(
        "Column '{name}' holds integers outside the 64-bit range, which cannot be used as \
         categories or a time index."
    ))
}

/// A float level's label: Rust's shortest round-trip form, R's "Inf"/"-Inf".
fn float_label(v: f64) -> String {
    if v == f64::INFINITY {
        "Inf".to_string()
    } else if v == f64::NEG_INFINITY {
        "-Inf".to_string()
    } else {
        v.to_string()
    }
}

/// "YYYY-MM-DD" of a polars Date (days since 1970-01-01).
fn date_label(days: i32) -> String {
    // 1970-01-01 is day 719 163 counted from 0001-01-01 (day 1).
    days.checked_add(719_163)
        .and_then(chrono::NaiveDate::from_num_days_from_ce_opt)
        .map_or_else(|| days.to_string(), |d| d.format("%Y-%m-%d").to_string())
}

/// The dtype rule of a time index.
fn check_time_dtype(col: &Column, name: &str) -> EconResult<()> {
    let dtype = col.dtype();
    if col.null_count() == col.len()
        || dtype.is_integer()
        || dtype.is_float()
        || dtype.is_decimal()
        || matches!(dtype, DataType::Date)
    {
        Ok(())
    } else {
        Err(invalid(format!(
            "Time column '{name}' must hold whole numbers or dates; it has type {dtype}."
        )))
    }
}

/// A time index over all rows (after [`check_time_dtype`]): `None` is missing.
fn time_values(col: &Column, name: &str) -> EconResult<Vec<Option<i64>>> {
    let n = col.len();
    if col.null_count() == n {
        return try_vec(n, None, "time index values");
    }
    let dtype = col.dtype();
    let mut out: Vec<Option<i64>> = try_with_capacity(n, "time index values")?;
    if dtype.is_signed_integer() {
        let cast = col
            .strict_cast(&DataType::Int64)
            .map_err(|_| integers_out_of_range(name))?;
        out.extend(cast.i64().map_err(polars_error)?.iter());
    } else if dtype.is_unsigned_integer() {
        let cast = col
            .strict_cast(&DataType::UInt64)
            .map_err(|_| integers_out_of_range(name))?;
        for (i, v) in cast.u64().map_err(polars_error)?.iter().enumerate() {
            out.push(match v {
                None => None,
                Some(v) => Some(i64::try_from(v).map_err(|_| {
                    invalid(format!(
                        "Time column '{name}': row {i} (0-based) is {v}, which is out of range \
                         for a time index."
                    ))
                })?),
            });
        }
    } else if matches!(dtype, DataType::Date) {
        let cast = col.strict_cast(&DataType::Int32).map_err(polars_error)?;
        out.extend(
            cast.i32()
                .map_err(polars_error)?
                .iter()
                .map(|v| v.map(i64::from)),
        );
    } else {
        // Float or Decimal. A whole number below 2^63 in magnitude converts exactly.
        const LIMIT: f64 = 9_223_372_036_854_775_808.0;
        let cast = col.strict_cast(&DataType::Float64).map_err(polars_error)?;
        for (i, v) in cast.f64().map_err(polars_error)?.iter().enumerate() {
            match v {
                None => out.push(None),
                Some(x) if x.is_nan() => out.push(None),
                Some(x) if x.fract() == 0.0 && x.abs() < LIMIT => out.push(Some(x as i64)),
                Some(x) => {
                    return Err(invalid(format!(
                        "Time column '{name}' must hold whole numbers; row {i} (0-based) is {x}. \
                         Round or recode it first."
                    )));
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use crate::resource;
    use p2a_validation::{Check, Tol};
    use polars::prelude::{
        Categories, DataType, FrozenCategories, IntoColumn, NamedFrom, Series, TimeUnit, df,
    };

    fn strings(labels: &[&str]) -> Vec<String> {
        labels.iter().map(|s| s.to_string()).collect()
    }

    fn frame(series: Series) -> DataFrame {
        DataFrame::new(vec![series.into_column()]).unwrap()
    }

    fn invalid_message(r: EconResult<impl std::fmt::Debug>) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_columns_conventions() {
        let (case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        // design 2.2: the fixture's dtypes are what the case relies on
        assert_eq!(df.column("y").unwrap().dtype(), &DataType::Float64);
        assert_eq!(df.column("n").unwrap().dtype(), &DataType::Int64);
        assert_eq!(df.column("s").unwrap().dtype(), &DataType::String);
        assert_eq!(df.column("flag").unwrap().dtype(), &DataType::Boolean);
        let cols = ColumnRequest::new(df)
            .numeric("y")
            .factor("n")
            .factor("s")
            .extract()
            .unwrap();
        let m = cols.missing();
        let by = |c: &str| {
            m.by_column
                .iter()
                .find(|(k, _)| k == c)
                .map_or(0, |(_, v)| *v) as f64
        };
        let code = |c: &Option<u32>| c.map_or(-1.0, f64::from);
        let (n_codes, n_levels) = factor_options(df, "n").unwrap();
        let (s_codes, s_levels) = factor_options(df, "s").unwrap();
        let flag: Vec<f64> = numeric_options(df, "flag")
            .unwrap()
            .iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        let y = cols.numeric("y");
        Check::new(&case)
            .vector(
                "complete_rows",
                &cols.rows().iter().map(|&r| r as f64).collect::<Vec<_>>(),
                Tol::exact(),
            )
            .scalar("n_complete", cols.n() as f64, Tol::exact())
            .scalar("n_dropped", m.n_dropped as f64, Tol::exact())
            .scalar("missing_y", by("y"), Tol::exact())
            .scalar("missing_n", by("n"), Tol::exact())
            .scalar("missing_s", by("s"), Tol::exact())
            .vector(
                "n_levels_numeric",
                &cols.factor("n").numeric_levels,
                Tol::exact(),
            )
            .vector(
                "n_codes",
                &n_codes.iter().map(code).collect::<Vec<_>>(),
                Tol::exact(),
            )
            .scalar("s_level_count", s_levels.len() as f64, Tol::exact())
            .vector(
                "s_codes",
                &s_codes.iter().map(code).collect::<Vec<_>>(),
                Tol::exact(),
            )
            .vector("flag_numeric", &flag, Tol::exact()) // null in the case = NaN here
            // six values; R's mean() accumulates in long double, so 1e-14 rather than exact
            .scalar(
                "mean_y_complete",
                y.iter().sum::<f64>() / y.len() as f64,
                Tol::new(1e-14, 0.0),
            )
            .finish();
        assert_eq!(n_levels, ["2", "4", "10", "33"]); // numeric, not string, order (ECON-19, STAT-50)
        assert_eq!(s_levels, ["A", "B", "a", "b", "c"]); // C collation; R in en_US.UTF-8 gives a A b B c
        assert_eq!(m.n_total, 10);
        assert_eq!(m.n_used, 6);
    }

    #[test]
    fn test_validate_columns_conventions_kept_rows() {
        let (case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        let cols = ColumnRequest::new(df)
            .numeric("y")
            .factor("n")
            .factor("s")
            .extract()
            .unwrap();
        // Every level occurs in the kept rows, so the kept-row codes are R's codes at
        // the complete rows.
        let rows = case.vector("complete_rows");
        let at = |key: &str| -> Vec<u32> {
            let all = case.vector(key);
            rows.iter().map(|&r| all[r as usize] as u32).collect()
        };
        assert_eq!(cols.factor("n").codes, at("n_codes"));
        assert_eq!(cols.factor("n").codes, [2, 0, 2, 0, 3, 1]);
        assert_eq!(cols.factor("s").codes, at("s_codes"));
        assert_eq!(cols.factor("s").codes, [3, 1, 4, 0, 4, 2]);
        assert_eq!(cols.factor("s").counts(), [1, 1, 1, 1, 2]);
        assert_eq!(cols.factor("s").label(0), "b");
        assert_eq!(cols.factor("s").n_levels(), 5);
        assert_eq!(cols.factor("n").levels, ["2", "4", "10", "33"]);
        assert_eq!(
            cols.factor("n").numeric_levels,
            case.vector("n_levels_numeric")
        );
        let id = ColumnRequest::new(df).factor("id").extract().unwrap();
        let expected: Vec<String> = (1..=10).map(|i| i.to_string()).collect();
        assert_eq!(id.factor("id").levels, expected); // "10" last: numeric order
    }

    #[test]
    fn test_validate_columns_fail_policy() {
        let (case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        match ColumnRequest::new(df)
            .numeric("y")
            .factor("n")
            .na(NaPolicy::Fail)
            .extract()
        {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "y");
                assert_eq!(count as f64, case.scalar("missing_y"));
            }
            other => panic!("expected NullValues for y, got {other:?}"),
        }
        match ColumnRequest::new(df)
            .factor("n")
            .numeric("y")
            .na(NaPolicy::Fail)
            .extract()
        {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "n");
                assert_eq!(count as f64, case.scalar("missing_n"));
            }
            other => panic!("expected NullValues for n, got {other:?}"),
        }
        let id = ColumnRequest::new(df)
            .numeric("id")
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        assert_eq!(id.n(), 10);
        assert!(id.missing().is_empty());
    }

    #[test]
    fn test_validate_columns_min_rows() {
        let (case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        let request = ColumnRequest::new(df).numeric("y").factor("n").factor("s");
        match request.clone().min_rows(7, "two-way test").extract() {
            Err(EconError::InsufficientData {
                required,
                provided,
                context,
            }) => {
                assert_eq!(required, 7);
                assert_eq!(provided as f64, case.scalar("n_complete"));
                assert!(
                    context.contains("two-way test") && context.contains("6 of 10"),
                    "{context}"
                );
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
        assert_eq!(
            request.min_rows(6, "two-way test").extract().unwrap().n(),
            6
        );
    }

    #[test]
    fn test_validate_columns_errors_name_the_column() {
        let (_case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        match ColumnRequest::new(df).numeric("yy").extract() {
            Err(EconError::ColumnNotFound { column, available }) => {
                assert_eq!(column, "yy");
                assert_eq!(available, ["id", "y", "n", "s", "flag"]);
            }
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        for request in [
            ColumnRequest::new(df).numeric("s"),
            ColumnRequest::new(df).numeric("flag").allow_bool(false),
        ] {
            match request.extract() {
                Err(EconError::NonNumericColumn { column }) => {
                    assert!(column == "s" || column == "flag", "{column}")
                }
                other => panic!("expected NonNumericColumn, got {other:?}"),
            }
        }
        let message = invalid_message(ColumnRequest::new(df).time_index("s").extract());
        assert!(message.contains("Time column 's'"), "{message}");
        assert!(matches!(
            column(df, "zz"),
            Err(EconError::ColumnNotFound { .. })
        ));
    }

    #[test]
    fn test_validate_columns_matrix_and_summary() {
        let (case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        let cols = ColumnRequest::new(df)
            .numerics(&["y", "n"])
            .extract()
            .unwrap();
        assert_eq!(cols.rows(), [0, 2, 4, 5, 7, 8, 9]);
        let m = cols.matrix(&["y", "n"], true).unwrap();
        assert_eq!(m.column_names, ["(Intercept)", "y", "n"]);
        assert_eq!(m.n_obs, 7);
        assert_eq!(m.n_features, 3);
        assert!(m.has_intercept);
        // Derived from the fixture, not case values (the case's complete rows are those of
        // complete.cases(y, n, s)): y sums to 44 and n to 94 over these 7 rows (binary-exact).
        let sums: Vec<f64> = (0..3).map(|j| m.data.column(j).sum()).collect();
        assert_eq!(sums, [7.0, 44.0, 94.0]);
        assert_eq!(cols.numeric_array("n").sum(), 94.0);
        // The per-column missing counts are the case's.
        assert_eq!(cols.missing().by_column.len(), 2);
        let count = |c: &str| {
            cols.missing()
                .by_column
                .iter()
                .find(|(k, _)| k == c)
                .map_or(0, |(_, v)| *v) as f64
        };
        Check::new(&case)
            .scalar("missing_y", count("y"), Tol::exact())
            .scalar("missing_n", count("n"), Tol::exact())
            .finish();
        let note = "3 of 10 rows dropped because of missing values (y: 2, n: 1)";
        assert_eq!(cols.missing().note().as_deref(), Some(note));
        assert_eq!(cols.missing().to_string(), note);

        let none = MissingSummary::none(10);
        assert!(none.is_empty());
        assert_eq!(none.note(), None);
        assert_eq!(none.to_string(), "");

        let merged = MissingSummary::merge_samples(
            &numeric_column(df, "y", NaPolicy::Omit).unwrap().1,
            &numeric_column(df, "n", NaPolicy::Omit).unwrap().1,
        );
        assert_eq!(
            merged,
            MissingSummary {
                n_total: 20,
                n_used: 17,
                n_dropped: 3,
                by_column: vec![("y".to_string(), 2), ("n".to_string(), 1)],
            }
        );

        #[derive(Serialize)]
        struct R {
            #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
            missing: MissingSummary,
        }
        let empty = serde_json::to_string(&R { missing: none }).unwrap();
        assert_eq!(empty, "{}");
        let full = serde_json::to_string(&R {
            missing: cols.missing().clone(),
        })
        .unwrap();
        // A streaming serializer writes by_column as an object in request order.
        assert_eq!(
            full,
            r#"{"missing":{"n_total":10,"n_used":7,"n_dropped":3,"by_column":{"y":2,"n":1}}}"#
        );
        let summary = r#"{"n_total":10,"n_used":7,"n_dropped":3,"by_column":{"y":2,"n":1}}"#;
        let back: MissingSummary = serde_json::from_str(summary).unwrap();
        assert_eq!(&back, cols.missing());
        assert_eq!(serde_json::to_string(&back).unwrap(), summary);
    }

    #[test]
    fn test_validate_columns_level_order() {
        let (_case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        let cols = ColumnRequest::new(df)
            .numeric("y")
            .factor("n")
            .factor("s")
            .extract()
            .unwrap();
        let s = cols.factor("s");
        let order = strings(&["c", "b", "a", "B", "A", "z"]);
        let reordered = s.with_level_order(&order).unwrap();
        assert_eq!(reordered.levels, order);
        assert_eq!(reordered.counts(), [2, 1, 1, 1, 1, 0]);
        assert_eq!(reordered.label(0), "b");
        assert!(reordered.numeric_levels.is_empty());

        let message = invalid_message(s.with_level_order(&strings(&["A", "B"])));
        assert!(message.contains("'a'"), "{message}");
        let message =
            invalid_message(s.with_level_order(&strings(&["A", "B", "a", "A", "b", "c"])));
        assert!(message.contains("twice"), "{message}");

        // Numeric levels follow the labels; a new label gets its parsed value.
        let n = cols
            .factor("n")
            .with_level_order(&strings(&["33", "10", "4", "2", "7"]))
            .unwrap();
        assert_eq!(n.numeric_levels, [33.0, 10.0, 4.0, 2.0, 7.0]);
        assert_eq!(n.label(0), "10");
        // A new label that is not a number has no numeric level: refused by name.
        for bad in ["ten", "NaN"] {
            let message = invalid_message(
                cols.factor("n")
                    .with_level_order(&strings(&["33", "10", "4", "2", "7", bad])),
            );
            assert!(message.contains(&format!("'{bad}'")), "{message}");
            assert!(message.contains("not a number"), "{message}");
        }
    }

    #[test]
    fn level_order_refusal_is_bounded() {
        // A caller-supplied level order can be long; the refusal names the unknown label
        // and the order's length, not every label of the order.
        let df = frame(Series::new("s".into(), &["a", "b", "a"]));
        let cols = ColumnRequest::new(&df).factor("s").extract().unwrap();
        let order: Vec<String> = (0..1000).map(|i| format!("L{i}")).collect();
        let message = invalid_message(cols.factor("s").with_level_order(&order));
        assert!(message.contains("'a'"), "{message}");
        assert!(message.contains("1000"), "{message}");
        assert!(!message.contains("L999"), "{message}");
        assert!(message.len() < 300, "{} bytes: {message}", message.len());
        // Each label is cut too: a long label in the order, a long observed label, and a long
        // label the order repeats.
        let long = "x".repeat(10_000);
        let order = vec![long.clone(), "b".to_string()];
        let message = invalid_message(cols.factor("s").with_level_order(&order));
        assert!(message.contains("'a'"), "{message}");
        assert!(message.len() < 300, "{} bytes", message.len());
        let wide = frame(Series::new("s".into(), &[long.as_str(), "b"]));
        let wide_cols = ColumnRequest::new(&wide).factor("s").extract().unwrap();
        let message = invalid_message(wide_cols.factor("s").with_level_order(&strings(&["b"])));
        assert!(message.len() < 300, "{} bytes", message.len());
        let twice = vec![long.clone(), long];
        let message = invalid_message(cols.factor("s").with_level_order(&twice));
        assert!(message.contains("twice"), "{message}");
        assert!(message.len() < 300, "{} bytes", message.len());
        // Five long multi-byte labels in the order: still bounded in bytes.
        let wide_order: Vec<String> = (0..1000)
            .map(|i| format!("{i}{}", "é".repeat(500)))
            .collect();
        let message = invalid_message(cols.factor("s").with_level_order(&wide_order));
        assert!(message.len() < 300, "{} bytes: {message}", message.len());
    }

    #[test]
    fn test_validate_columns_duplicate_names() {
        let (case, ds) = crate::test_support::load_case("columns_conventions");
        let df = ds.df();
        let cols = ColumnRequest::new(df)
            .numerics(&["y", "y"])
            .numeric("y")
            .extract()
            .unwrap();
        assert_eq!(cols.missing().by_column.len(), 1);
        assert_eq!(cols.missing().by_column[0].0, "y");
        assert_eq!(cols.n(), 8);
        assert_eq!(cols.matrix(&["y", "y"], false).unwrap().data.dim(), (8, 2));
        // The same name in two roles is extracted in both and counted once.
        let both = ColumnRequest::new(df)
            .numeric("n")
            .factor("n")
            .extract()
            .unwrap();
        assert_eq!(both.missing().by_column.len(), 1);
        assert_eq!(both.missing().by_column[0].0, "n");
        assert_eq!(both.numeric("n").len(), 9);
        assert_eq!(both.factor("n").codes.len(), 9);
        // A repeated name is counted once: the counts are the case's per-column counts.
        Check::new(&case)
            .scalar(
                "missing_y",
                cols.missing().by_column[0].1 as f64,
                Tol::exact(),
            )
            .scalar(
                "missing_n",
                both.missing().by_column[0].1 as f64,
                Tol::exact(),
            )
            .finish();
    }

    #[test]
    fn test_na_columns_numeric_dtypes() {
        let mut df = df! {
            "i8" => [1i8, 2, -3, 4],
            "i16" => [1i16, 2, -3, 4],
            "i32" => [1i32, 2, -3, 4],
            "i64" => [1i64, 2, -3, 4],
            "u8" => [1u8, 2, 3, 4],
            "u16" => [1u16, 2, 3, 4],
            "u32" => [1u32, 2, 3, 4],
            "u64" => [1u64, 2, 3, 4],
            "f32" => [1.25f32, 2.5, -3.75, 4.0],
            "b" => [true, false, true, true],
        }
        .unwrap();
        let dec = Series::new("dec".into(), ["1.25", "2.50", "-3.75", "4.00"])
            .strict_cast(&DataType::Decimal(10, 2))
            .unwrap();
        df.with_column(dec).unwrap();
        let expected: [(&str, [f64; 4]); 11] = [
            ("i8", [1.0, 2.0, -3.0, 4.0]),
            ("i16", [1.0, 2.0, -3.0, 4.0]),
            ("i32", [1.0, 2.0, -3.0, 4.0]),
            ("i64", [1.0, 2.0, -3.0, 4.0]),
            ("u8", [1.0, 2.0, 3.0, 4.0]),
            ("u16", [1.0, 2.0, 3.0, 4.0]),
            ("u32", [1.0, 2.0, 3.0, 4.0]),
            ("u64", [1.0, 2.0, 3.0, 4.0]),
            ("f32", [1.25, 2.5, -3.75, 4.0]),
            ("dec", [1.25, 2.5, -3.75, 4.0]),
            ("b", [1.0, 0.0, 1.0, 1.0]),
        ];
        for (name, values) in expected {
            let cols = ColumnRequest::new(&df).numeric(name).extract().unwrap();
            assert_eq!(cols.numeric(name), values, "{name}");
            let dm = DesignMatrix::from_dataframe(&df, &[name], false).unwrap();
            assert_eq!(dm.data.column(0).to_vec(), values, "{name} (DesignMatrix)");
            assert!(
                is_numeric_dtype(df.column(name).unwrap().dtype(), true),
                "{name}"
            );
        }
        for dtype in [
            DataType::Int128,
            DataType::UInt128,
            DataType::Decimal(10, 2),
        ] {
            assert!(is_numeric_dtype(&dtype, false), "{dtype}");
        }
        assert!(is_numeric_dtype(&DataType::Boolean, true));
        assert!(!is_numeric_dtype(&DataType::Boolean, false));
        for dtype in [
            DataType::String,
            DataType::Date,
            DataType::Datetime(TimeUnit::Milliseconds, None),
            DataType::from_categories(Categories::global()),
            DataType::List(Box::new(DataType::Int64)),
        ] {
            assert!(!is_numeric_dtype(&dtype, true), "{dtype}");
        }
    }

    #[test]
    fn test_na_columns_nan_is_missing_inf_is_refused() {
        let df = df! {
            "y" => [1.0, f64::NAN, 3.0, 4.5, 5.25],
            "x" => [Some(1.0), Some(2.0), None, Some(f64::INFINITY), Some(5.0)],
            "x2" => [1.0, f64::INFINITY, 3.0, 4.0, 5.0],
        }
        .unwrap();
        match ColumnRequest::new(&df).numeric("y").numeric("x").extract() {
            Err(EconError::NonFiniteValues { column, count }) => {
                assert_eq!(column, "x");
                assert_eq!(count, 1);
            }
            other => panic!("expected NonFiniteValues, got {other:?}"),
        }
        let kept = ColumnRequest::new(&df)
            .numeric("y")
            .numeric("x")
            .allow_infinite(true)
            .extract()
            .unwrap();
        assert_eq!(kept.rows(), [0, 3, 4]);
        assert_eq!(kept.numeric("x"), [1.0, f64::INFINITY, 5.0]);
        // The Inf of x2 sits in a row that y's NaN already drops (R: nobs 4).
        let ok = ColumnRequest::new(&df)
            .numeric("y")
            .numeric("x2")
            .extract()
            .unwrap();
        assert_eq!(ok.rows(), [0, 2, 3, 4]);
        assert_eq!(
            numeric_options(&df, "y").unwrap(),
            [Some(1.0), None, Some(3.0), Some(4.5), Some(5.25)]
        );
        assert!(matches!(
            numeric_options(&df, "x"),
            Err(EconError::NonFiniteValues { count: 1, .. })
        ));
        match ColumnRequest::new(&df)
            .numeric("y")
            .na(NaPolicy::Fail)
            .extract()
        {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "y");
                assert_eq!(count, 1);
            }
            other => panic!("expected NullValues, got {other:?}"),
        }
    }

    #[test]
    fn test_na_columns_factor_dtypes() {
        let ints = frame(Series::new(
            "g".into(),
            [Some(10i64), Some(2), Some(33), Some(2), None],
        ));
        let cols = ColumnRequest::new(&ints).factor("g").extract().unwrap();
        let g = cols.factor("g");
        assert_eq!(g.levels, ["2", "10", "33"]);
        assert_eq!(g.codes, [1, 0, 2, 0]);
        assert_eq!(g.numeric_levels, [2.0, 10.0, 33.0]);
        assert_eq!(cols.n(), 4);

        let floats = frame(Series::new(
            "g".into(),
            [1.5, -2.0, -0.0, 0.0, f64::NAN, f64::INFINITY],
        ));
        let cols = ColumnRequest::new(&floats).factor("g").extract().unwrap();
        // signed zero is one level, as in R
        assert_eq!(cols.factor("g").levels, ["-2", "0", "1.5", "Inf"]);
        assert_eq!(cols.n(), 5);

        let bools = frame(Series::new("g".into(), [Some(true), Some(false), None]));
        let (_, levels) = factor_options(&bools, "g").unwrap();
        assert_eq!(levels, ["FALSE", "TRUE"]);

        let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let day = |y, m, d| {
            chrono::NaiveDate::from_ymd_opt(y, m, d)
                .unwrap()
                .signed_duration_since(epoch)
                .num_days() as i32
        };
        let dates = frame(
            Series::new(
                "g".into(),
                [day(2001, 3, 1), day(2000, 12, 31), day(2001, 3, 1)],
            )
            .cast(&DataType::Date)
            .unwrap(),
        );
        let (codes, levels) = factor_options(&dates, "g").unwrap();
        assert_eq!(levels, ["2000-12-31", "2001-03-01"]);
        assert_eq!(codes, [Some(1), Some(0), Some(1)]);

        let cats = frame(
            Series::new("g".into(), ["b", "a", "b"])
                .cast(&DataType::from_categories(Categories::global()))
                .unwrap(),
        );
        assert_eq!(factor_options(&cats, "g").unwrap().1, ["a", "b"]);

        let declared = FrozenCategories::new(["low", "mid", "high"]).unwrap();
        let enums = frame(
            Series::new("g".into(), ["high", "low", "high"])
                .cast(&DataType::from_frozen_categories(declared))
                .unwrap(),
        );
        let (codes, levels) = factor_options(&enums, "g").unwrap();
        assert_eq!(levels, ["low", "high"]); // declared order, observed levels only
        assert_eq!(codes, [Some(1), Some(0), Some(1)]);

        let lists = frame(Series::new(
            "lst".into(),
            [
                Series::new("".into(), [1i64, 2]),
                Series::new("".into(), [3i64]),
            ],
        ));
        let message = invalid_message(ColumnRequest::new(&lists).factor("lst").extract());
        assert!(message.contains("'lst'"), "{message}");
    }

    #[test]
    fn test_na_columns_time_index() {
        let ints = frame(Series::new(
            "year".into(),
            [Some(2001i64), Some(2000), None],
        ));
        let cols = ColumnRequest::new(&ints)
            .time_index("year")
            .extract()
            .unwrap();
        assert_eq!(cols.time_index("year"), [2001, 2000]);
        assert_eq!(cols.missing().n_dropped, 1);

        let whole = frame(Series::new("year".into(), [2000.0, 2001.0]));
        let cols = ColumnRequest::new(&whole)
            .time_index("year")
            .extract()
            .unwrap();
        assert_eq!(cols.time_index("year"), [2000, 2001]);

        let fractional = frame(Series::new("year".into(), [2000.0, 2000.25]));
        assert_eq!(
            invalid_message(ColumnRequest::new(&fractional).time_index("year").extract()),
            "Time column 'year' must hold whole numbers; row 1 (0-based) is 2000.25. \
             Round or recode it first."
        );

        let dates = frame(
            Series::new("year".into(), [10957i32, 10958])
                .cast(&DataType::Date)
                .unwrap(),
        );
        let cols = ColumnRequest::new(&dates)
            .time_index("year")
            .extract()
            .unwrap();
        assert_eq!(cols.time_index("year"), [10957, 10958]);

        let text = frame(Series::new("year".into(), ["2000", "2001"]));
        let message = invalid_message(ColumnRequest::new(&text).time_index("year").extract());
        assert!(
            message.contains("Time column 'year' must hold whole numbers or dates"),
            "{message}"
        );

        let huge = frame(Series::new("year".into(), [u64::MAX]));
        let message = invalid_message(ColumnRequest::new(&huge).time_index("year").extract());
        assert!(message.contains("out of range"), "{message}");
    }

    #[test]
    fn test_na_columns_all_null_column() {
        let df = DataLoader::from_csv_string("id,b\n1,\n2,\n3,\n").unwrap();
        for request in [
            ColumnRequest::new(&df).numeric("b"),
            ColumnRequest::new(&df).factor("b"),
            ColumnRequest::new(&df).time_index("b"),
        ] {
            match request.extract() {
                Err(EconError::InsufficientData {
                    required, provided, ..
                }) => {
                    assert_eq!((required, provided), (1, 0));
                }
                other => panic!("expected InsufficientData, got {other:?}"),
            }
        }
        match ColumnRequest::new(&df)
            .numeric("b")
            .na(NaPolicy::Fail)
            .extract()
        {
            Err(EconError::NullValues { column, count }) => {
                assert_eq!(column, "b");
                assert_eq!(count, 3);
            }
            other => panic!("expected NullValues, got {other:?}"),
        }
    }

    #[test]
    fn test_na_columns_allocation_is_fallible() {
        let y: Vec<f64> = (0..1000)
            .map(|i| i as f64 * 0.5 + ((i * 7) % 11) as f64 * 0.1)
            .collect();
        let df = df! { "y" => y }.unwrap();
        let r = resource::with_alloc_ceiling(Some(64), || {
            ColumnRequest::new(&df).numeric("y").extract()
        });
        assert!(matches!(r, Err(EconError::ResourceLimit(_))), "{r:?}");
    }

    #[test]
    fn test_na_columns_two_sample_note() {
        // Two samples of 22 values, each dropping its own missing values, merged as the
        // two-sample wrappers merge them: the note counts values, and names the sample size.
        let sample = |name: &str, dropped: usize| MissingSummary {
            n_total: 22,
            n_used: 22 - dropped,
            n_dropped: dropped,
            by_column: vec![(name.to_string(), dropped)],
        };
        let merged = MissingSummary::merge_samples(&sample("x", 15), &sample("y", 15));
        assert_eq!(
            merged.two_sample_note().as_deref(),
            Some(
                "30 of 44 values dropped because of missing values (two samples of 22; x: 15, \
                 y: 15)"
            )
        );
        // note() is unchanged.
        assert_eq!(
            merged.note().as_deref(),
            Some("30 of 44 rows dropped because of missing values (x: 15, y: 15)")
        );
        // Only one sample with missing values.
        let one = MissingSummary::merge_samples(&sample("xi", 2), &MissingSummary::none(22));
        assert_eq!(
            one.two_sample_note().as_deref(),
            Some("2 of 44 values dropped because of missing values (two samples of 22; xi: 2)")
        );
        // One column as both samples: its values are counted in each sample.
        let same = MissingSummary::merge_samples(&sample("x", 15), &sample("x", 15));
        assert_eq!(
            same.two_sample_note().as_deref(),
            Some("30 of 44 values dropped because of missing values (two samples of 22; x: 30)")
        );
        let none =
            MissingSummary::merge_samples(&MissingSummary::none(5), &MissingSummary::none(5));
        assert_eq!(none.two_sample_note(), None);
    }

    #[test]
    fn test_na_columns_matrix_allocation_is_fallible() {
        // The extraction holds one column; a design matrix that repeats it many times does not
        // fit the same ceiling, and is refused instead of allocated.
        let y: Vec<f64> = (0..64)
            .map(|i| i as f64 * 0.5 + ((i * 7) % 11) as f64 * 0.1)
            .collect();
        let df = df! { "y" => y }.unwrap();
        let cols = ColumnRequest::new(&df).numeric("y").extract().unwrap();
        let repeated = vec!["y"; 64];
        match resource::with_alloc_ceiling(Some(4096), || cols.matrix(&repeated, true)) {
            Err(EconError::ResourceLimit(message)) => {
                assert!(message.contains("design matrix"), "{message}")
            }
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        let small = resource::with_alloc_ceiling(Some(4096), || cols.matrix(&["y"], true)).unwrap();
        assert_eq!(small.data.dim(), (64, 2));
        assert_eq!(small.data.column(1).sum(), cols.numeric_array("y").sum());
    }

    #[test]
    fn test_econ_error_from_design_maps_every_variant() {
        let df = df! { "a" => [1.0, 2.0], "b" => [0.5, 1.5] }.unwrap();
        match econ_error_from_design(DesignError::ColumnNotFound("q".into()), &df) {
            EconError::ColumnNotFound { column, available } => {
                assert_eq!(column, "q");
                assert_eq!(available, ["a", "b"]);
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            econ_error_from_design(DesignError::NonNumericColumn("a".into()), &df),
            EconError::NonNumericColumn { column } if column == "a"
        ));
        let null = DesignError::NullValues {
            column: "a".into(),
            count: 7,
            first_rows: vec![0, 3, 4, 5, 9],
        };
        assert!(matches!(
            econ_error_from_design(null, &df),
            EconError::NullValues { column, count: 7 } if column == "a"
        ));
        let inf = DesignError::NonFinite {
            column: "b".into(),
            count: 2,
            first_rows: vec![1, 6],
        };
        assert!(matches!(
            econ_error_from_design(inf, &df),
            EconError::NonFiniteValues { column, count: 2 } if column == "b"
        ));
        assert!(matches!(
            econ_error_from_design(DesignError::EmptyDataset, &df),
            EconError::EmptyDataset
        ));
        assert!(matches!(
            econ_error_from_design(DesignError::NoColumns, &df),
            EconError::InvalidSpecification { message }
                if message == "No independent variables specified"
        ));
        assert!(matches!(
            econ_error_from_design(
                DesignError::PolarsError(PolarsError::ComputeError("x".into())),
                &df
            ),
            EconError::Internal(_)
        ));
        assert!(matches!(
            econ_error_from_design(DesignError::ResourceLimit("too large".into()), &df),
            EconError::ResourceLimit(message) if message == "too large"
        ));
    }
}
