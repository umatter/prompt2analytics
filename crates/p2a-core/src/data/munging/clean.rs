//! Data cleaning operations: drop_na, fill_na, deduplicate, cast, replace, regex.

use polars::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::CHECKPOINT_EVERY;
use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use crate::data::columns::is_numeric_dtype;

/// Strategy for handling missing values when filling.
///
/// `Constant` and `Zero` write a value of the column's own type, so the column keeps its type
/// (`Mean` and `Median` give Float64). `FillStrategy::parse` reads the strategy from user input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FillStrategy {
    /// Fill with a constant, parsed in the column's type: every integer width (with a range
    /// check), Float32/64, Boolean (as R's `as.logical`), Date (`YYYY-MM-DD`), String (verbatim),
    /// Categorical or Enum labels.
    Constant(String),
    /// Fill with the mean of the observed values (numeric only)
    Mean,
    /// Fill with the median of the observed values (numeric only)
    Median,
    /// Forward fill (use previous non-null value)
    Forward,
    /// Backward fill (use next non-null value)
    Backward,
    /// Fill with zero (numeric), FALSE (boolean) or an empty string (text).
    Zero,
    /// The most frequent observed value, R `names(which.max(table(x)))`: ties go to the first level
    /// in R's factor order (numbers ascending, text by bytes, FALSE before TRUE, dates
    /// chronological), not to the first occurrence.
    Mode,
}

/// The strategy keyword, for parsing user input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillStrategyKind {
    Mean,
    Median,
    Mode,
    Forward,
    Backward,
    Zero,
    Constant,
}

crate::option_enum!(
    FillStrategyKind,
    crate::options::Choices {
        option: "fill strategy",
        values: &[
            (&["mean"], FillStrategyKind::Mean),
            (&["median"], FillStrategyKind::Median),
            (&["mode", "most_frequent"], FillStrategyKind::Mode),
            (
                &["forward", "ffill", "locf", "forward_fill"],
                FillStrategyKind::Forward
            ),
            (
                &["backward", "bfill", "nocb", "backward_fill"],
                FillStrategyKind::Backward
            ),
            (&["zero"], FillStrategyKind::Zero),
            (&["constant", "value"], FillStrategyKind::Constant),
        ],
    }
);

impl FillStrategy {
    /// Parse a strategy keyword (case- and separator-insensitive) and, for `constant`, take the fill
    /// value from `value` with its case kept.
    ///
    /// # Errors
    ///
    /// [`MungeError::InvalidExpression`] for an unknown keyword, `constant` without a value, and a
    /// value with another strategy.
    pub fn parse(strategy: &str, value: Option<&str>) -> MungeResult<FillStrategy> {
        let kind: FillStrategyKind = strategy.parse().map_err(|e| match e {
            crate::errors::EconError::InvalidSpecification { message } => {
                MungeError::InvalidExpression(message)
            }
            other => MungeError::InvalidExpression(other.to_string()),
        })?;
        if kind == FillStrategyKind::Constant {
            return match value {
                Some(v) => Ok(FillStrategy::Constant(v.to_string())),
                None => Err(MungeError::InvalidExpression(
                    "fill strategy 'constant' needs a value".to_string(),
                )),
            };
        }
        if value.is_some() {
            return Err(MungeError::InvalidExpression(format!(
                "a fill value is only used with strategy 'constant', not '{kind}'"
            )));
        }
        Ok(match kind {
            FillStrategyKind::Mean => FillStrategy::Mean,
            FillStrategyKind::Median => FillStrategy::Median,
            FillStrategyKind::Mode => FillStrategy::Mode,
            FillStrategyKind::Forward => FillStrategy::Forward,
            FillStrategyKind::Backward => FillStrategy::Backward,
            FillStrategyKind::Zero => FillStrategy::Zero,
            FillStrategyKind::Constant => unreachable!("returned above"),
        })
    }

    /// The keyword of this strategy.
    pub fn kind(&self) -> FillStrategyKind {
        match self {
            FillStrategy::Constant(_) => FillStrategyKind::Constant,
            FillStrategy::Mean => FillStrategyKind::Mean,
            FillStrategy::Median => FillStrategyKind::Median,
            FillStrategy::Forward => FillStrategyKind::Forward,
            FillStrategy::Backward => FillStrategyKind::Backward,
            FillStrategy::Zero => FillStrategyKind::Zero,
            FillStrategy::Mode => FillStrategyKind::Mode,
        }
    }
}

/// What [`fill_na_with`] did.
#[derive(Clone)]
pub struct FillOutcome {
    pub dataset: Dataset,
    /// Every column the strategy was applied to, in the order filled, with the number of cells
    /// filled (0 allowed).
    pub filled: Vec<(String, usize)>,
    /// The value written into each filled column, as text: the constant, "0"/"false"/"", the mean or
    /// median (`{}` of the f64), the mode's label; absent for forward/backward.
    pub fill_values: Vec<(String, String)>,
    /// Columns skipped because no columns were named and the strategy does not apply to their type
    /// (column, the error a named column would have raised, as text).
    pub skipped: Vec<(String, String)>,
}

impl FillOutcome {
    /// One sentence for the user: "Filled 6 missing values in 3 columns: k (2, with 1), f (2, with
    /// 2.5), s (2, with 'a')". Text values are quoted, forward and backward print "k (2)", and only
    /// columns with at least one filled cell are listed; "No missing values were filled" when none
    /// was; then, if any column was skipped, "; skipped 2 columns: age (reason), when (reason)".
    pub fn summary(&self) -> String {
        fn count(n: usize, one: &str) -> String {
            if n == 1 {
                format!("1 {one}")
            } else {
                format!("{n} {one}s")
            }
        }
        let changed: Vec<&(String, usize)> = self.filled.iter().filter(|(_, n)| *n > 0).collect();
        let mut out = if changed.is_empty() {
            "No missing values were filled".to_string()
        } else {
            let total = changed.iter().map(|(_, n)| *n).sum::<usize>();
            let parts: Vec<String> = changed
                .iter()
                .map(|(c, n)| {
                    let value = self.fill_values.iter().find(|(v, _)| v == c);
                    match value {
                        Some((_, v)) if self.is_text_column(c) => format!("{c} ({n}, with '{v}')"),
                        Some((_, v)) => format!("{c} ({n}, with {v})"),
                        None => format!("{c} ({n})"),
                    }
                })
                .collect();
            format!(
                "Filled {} in {}: {}",
                count(total, "missing value"),
                count(changed.len(), "column"),
                parts.join(", ")
            )
        };
        if !self.skipped.is_empty() {
            let parts: Vec<String> = self
                .skipped
                .iter()
                .map(|(c, reason)| format!("{c} ({reason})"))
                .collect();
            out.push_str(&format!(
                "; skipped {}: {}",
                count(self.skipped.len(), "column"),
                parts.join(", ")
            ));
        }
        out
    }

    fn is_text_column(&self, column: &str) -> bool {
        self.dataset.df().column(column).is_ok_and(|c| {
            matches!(
                c.dtype(),
                DataType::String | DataType::Categorical(..) | DataType::Enum(..)
            )
        })
    }
}

/// Drop rows containing null values.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to check for nulls (None = all columns)
/// * `how` - "any" (drop if any null) or "all" (drop only if all specified columns are null)
///
/// # Example
/// ```ignore
/// // Drop rows with any null
/// let clean = drop_na(&dataset, None, "any")?;
///
/// // Drop rows with null in specific columns
/// let clean = drop_na(&dataset, Some(&["age", "income"]), "any")?;
/// ```
pub fn drop_na(dataset: &Dataset, columns: Option<&[&str]>, how: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns if specified
    if let Some(cols) = columns {
        for col in cols {
            if df.column(col).is_err() {
                return Err(MungeError::ColumnNotFound(col.to_string()));
            }
        }
    }

    let cols_to_check: Vec<&str> = match columns {
        Some(cols) => cols.to_vec(),
        None => df
            .get_column_names()
            .into_iter()
            .map(|s| s.as_str())
            .collect(),
    };

    // Build null mask
    let mut null_mask: Option<BooleanChunked> = None;

    for col_name in &cols_to_check {
        let col = df.column(col_name)?;
        let is_null = col.is_null();

        null_mask = Some(match null_mask {
            None => is_null,
            Some(existing) => match how.to_lowercase().as_str() {
                "any" => existing | is_null,
                "all" => existing & is_null,
                _ => {
                    return Err(MungeError::InvalidExpression(format!(
                        "Invalid 'how' parameter: {}. Use 'any' or 'all'",
                        how
                    )));
                }
            },
        });
    }

    // Invert mask to keep non-null rows
    let keep_mask = match null_mask {
        Some(mask) => !mask,
        None => return Ok(dataset.clone()),
    };

    let filtered = df.filter(&keep_mask)?;
    Ok(Dataset::new(filtered))
}

/// Fill missing values in specified columns.
///
/// A value is missing when it is null or, in a float column, NaN (R's `is.na`;
/// `tidyr::replace_na` fills NaN too), so every strategy fills NaN cells.
/// Mean and median are those of the observed values; integer columns become
/// Float64 for them.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to fill (None = all columns; then a column the strategy does not apply
///   to is skipped)
/// * `strategy` - How to fill missing values
///
/// # Errors
///
/// As [`fill_na_with`].
///
/// # Example
/// ```ignore
/// // Fill all numeric nulls with mean
/// let filled = fill_na(&dataset, None, FillStrategy::Mean)?;
///
/// // Fill specific column with constant
/// let filled = fill_na(&dataset, Some(&["age"]), FillStrategy::Constant("0".to_string()))?;
/// ```
pub fn fill_na(
    dataset: &Dataset,
    columns: Option<&[&str]>,
    strategy: FillStrategy,
) -> MungeResult<Dataset> {
    fill_na_with(dataset, columns, &strategy).map(|o| o.dataset)
}

/// Fill missing values and report what was filled ([`fill_na`] returns only the dataset).
///
/// With no columns named, a column the strategy or the constant does not fit (a text constant and a
/// numeric column, mean of a text column, zero of a date) is left as it is and listed in
/// [`FillOutcome::skipped`]; a named one is an error.
///
/// # Errors
///
/// [`MungeError::ColumnNotFound`] for a named column that does not exist;
/// [`MungeError::TypeMismatch`] for a named column whose type the strategy does not apply to (mean
/// or median of a non-numeric column; zero of a date or categorical column; any strategy but
/// forward and backward on a datetime, time, duration, decimal, binary or nested column);
/// [`MungeError::InvalidValue`] for a constant the column's type cannot hold (named columns), and,
/// named or not, for the mean, median or mode of a column without an observed value, which is
/// undefined.
pub fn fill_na_with(
    dataset: &Dataset,
    columns: Option<&[&str]>,
    strategy: &FillStrategy,
) -> MungeResult<FillOutcome> {
    let df = dataset.df();

    let cols_to_fill: Vec<String> = match columns {
        Some(cols) => {
            for col in cols {
                if df.column(col).is_err() {
                    return Err(MungeError::ColumnNotFound(col.to_string()));
                }
            }
            cols.iter().map(|s| s.to_string()).collect()
        }
        None => df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };

    let named = columns.is_some();
    let mut result_df = df.clone();
    let mut filled = Vec::new();
    let mut fill_values = Vec::new();
    let mut skipped = Vec::new();

    for col_name in &cols_to_fill {
        let source = result_df.column(col_name)?;
        let dtype = source.dtype().clone();
        let col = &nan_to_null(source)?;
        let missing_before = col.null_count();

        let outcome = match strategy {
            FillStrategy::Constant(value) => fill_constant(col, value)?,
            FillStrategy::Mean | FillStrategy::Median => {
                if !named && !is_numeric_dtype(&dtype, false) {
                    // Nobody named this non-numeric column.
                    ColumnFill::Inapplicable(MungeError::TypeMismatch {
                        column: col_name.clone(),
                        expected: "numeric (fill_na mean or median)".to_string(),
                        found: dtype.to_string(),
                    })
                } else {
                    let median = matches!(strategy, FillStrategy::Median);
                    let (column, value) = fill_with_statistic(&result_df, col_name, col, median)?;
                    ColumnFill::Filled {
                        column,
                        value: Some(value.to_string()),
                    }
                }
            }
            FillStrategy::Forward => ColumnFill::Filled {
                column: fill_forward(col)?,
                value: None,
            },
            FillStrategy::Backward => ColumnFill::Filled {
                column: fill_backward(col)?,
                value: None,
            },
            FillStrategy::Zero => fill_zero(col)?,
            FillStrategy::Mode => fill_mode(&result_df, col_name, col)?,
        };

        match outcome {
            ColumnFill::Filled { column, value } => {
                filled.push((
                    col_name.clone(),
                    missing_before.saturating_sub(column.null_count()),
                ));
                if let Some(value) = value {
                    fill_values.push((col_name.clone(), value));
                }
                result_df.with_column(column)?;
            }
            ColumnFill::Inapplicable(e) => {
                if named {
                    return Err(e);
                }
                skipped.push((col_name.clone(), e.to_string()));
            }
        }
    }

    Ok(FillOutcome {
        dataset: Dataset::new(result_df),
        filled,
        fill_values,
        skipped,
    })
}

/// What a fill does to one column.
enum ColumnFill {
    /// The filled column and the value written, as text (forward and backward have none).
    Filled {
        column: Column,
        value: Option<String>,
    },
    /// This strategy or value does not fit this column's type: an error when the column was named, a
    /// `skipped` entry when it was not.
    Inapplicable(MungeError),
}

/// A literal that parsed gives `Ok(value)`; one that did not is `Inapplicable`, not a failed fill.
macro_rules! parsed_or_inapplicable {
    ($parsed:expr) => {
        match $parsed {
            Ok(v) => v,
            Err(e) => return Ok(ColumnFill::Inapplicable(e)),
        }
    };
}

/// The missing cells of `col` (NaN already null) set to the constant `text`, parsed in the column's
/// type.
fn fill_constant(col: &Column, text: &str) -> MungeResult<ColumnFill> {
    let dtype = col.dtype();
    let name = col.name().clone();
    let column = col.name().as_str();

    macro_rules! integer {
        ($method:ident, $native:ty) => {{
            let v = parsed_or_inapplicable!(parse_int::<$native>(column, dtype, text));
            col.$method()?.fill_null_with_values(v)?.into_column()
        }};
    }

    let filled = match dtype {
        DataType::Int8 => integer!(i8, i8),
        DataType::Int16 => integer!(i16, i16),
        DataType::Int32 => integer!(i32, i32),
        DataType::Int64 => integer!(i64, i64),
        DataType::Int128 => integer!(i128, i128),
        DataType::UInt8 => integer!(u8, u8),
        DataType::UInt16 => integer!(u16, u16),
        DataType::UInt32 => integer!(u32, u32),
        DataType::UInt64 => integer!(u64, u64),
        DataType::UInt128 => integer!(u128, u128),
        DataType::Float64 => {
            let v = parsed_or_inapplicable!(parse_float(column, dtype, text));
            // na-ok: dtype matched by the arm; fill_na mapped NaN to null
            col.f64()?.fill_null_with_values(v)?.into_column()
        }
        DataType::Float32 => {
            let v = parsed_or_inapplicable!(parse_f32(column, dtype, text));
            // na-ok: dtype matched by the arm; fill_na mapped NaN to null
            col.f32()?.fill_null_with_values(v)?.into_column()
        }
        DataType::Boolean => {
            let v = parsed_or_inapplicable!(parse_bool(column, text));
            col.bool()?.fill_null_with_values(v)?.into_column()
        }
        DataType::Date => {
            let v = parsed_or_inapplicable!(parse_date(column, text));
            col.date()?
                .physical()
                .fill_null_with_values(v)?
                .into_date()
                .into_column()
        }
        DataType::String => fill_text(col.str()?, text).with_name(name).into_column(),
        DataType::Categorical(..) | DataType::Enum(..) => {
            let as_text = col.cast(&DataType::String)?;
            let filled = fill_text(as_text.str()?, text)
                .with_name(name)
                .into_column();
            parsed_or_inapplicable!(filled.strict_cast(dtype).map_err(|_| invalid_value(
                column,
                text,
                "not a category of the enum column".to_string()
            )))
        }
        other => {
            return Ok(ColumnFill::Inapplicable(MungeError::TypeMismatch {
                column: column.to_string(),
                expected: "a string, categorical, boolean, integer, float or date column"
                    .to_string(),
                found: other.to_string(),
            }));
        }
    };
    Ok(ColumnFill::Filled {
        column: filled,
        value: Some(text.to_string()),
    })
}

fn fill_text(ca: &StringChunked, value: &str) -> StringChunked {
    ca.iter().map(|cell| Some(cell.unwrap_or(value))).collect()
}

/// The missing cells of `col` set to 0, FALSE or "": R's `tidyr::replace_na(x, 0)` has no zero for a
/// date or a factor, and neither has this.
fn fill_zero(col: &Column) -> MungeResult<ColumnFill> {
    let dtype = col.dtype();
    let (text, printed) = match dtype {
        dt if dt.is_integer() || dt.is_float() => ("0", "0"),
        DataType::Boolean => ("FALSE", "false"),
        DataType::String => ("", ""),
        other => {
            return Ok(ColumnFill::Inapplicable(MungeError::TypeMismatch {
                column: col.name().to_string(),
                expected: "a numeric, boolean or string column (fill_na zero)".to_string(),
                found: other.to_string(),
            }));
        }
    };
    Ok(match fill_constant(col, text)? {
        ColumnFill::Filled { column, .. } => ColumnFill::Filled {
            column,
            value: Some(printed.to_string()),
        },
        inapplicable => inapplicable,
    })
}

/// The missing cells of `col` (`name` of `df`, NaN already null in `col`) set to the most frequent
/// observed value, as R's `names(which.max(table(x)))`: ties go to the first level in R's factor
/// order. A column without an observed value is an error.
fn fill_mode(df: &DataFrame, name: &str, col: &Column) -> MungeResult<ColumnFill> {
    let dtype = col.dtype();
    let supported = dtype.is_integer()
        || dtype.is_float()
        || matches!(
            dtype,
            DataType::String
                | DataType::Categorical(..)
                | DataType::Enum(..)
                | DataType::Boolean
                | DataType::Date
        );
    if !supported {
        return Ok(ColumnFill::Inapplicable(MungeError::TypeMismatch {
            column: name.to_string(),
            expected: "a string, categorical, boolean, integer, float or date column".to_string(),
            found: dtype.to_string(),
        }));
    }
    let stopped = |e: crate::errors::EconError| MungeError::FeatureError(e.to_string());
    let (codes, levels) = crate::data::columns::factor_options(df, name).map_err(stopped)?;
    let mut counts =
        crate::resource::try_vec(levels.len(), 0usize, "mode level counts").map_err(stopped)?;
    for (i, code) in codes.iter().enumerate() {
        if i.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(stopped)?;
        }
        if let Some(code) = code {
            counts[*code as usize] += 1;
        }
    }
    // The first index holding the maximum, as which.max.
    let most = counts
        .iter()
        .enumerate()
        .fold(None::<(usize, usize)>, |best, (i, &n)| match best {
            Some((_, m)) if m >= n => best,
            _ if n > 0 => Some((i, n)),
            _ => best,
        });
    let Some((level, _)) = most else {
        return Err(invalid_value(
            name,
            "mode",
            "the column has no observed values (all null or NaN), so its mode is undefined; \
             drop the column or fill it with a constant"
                .to_string(),
        ));
    };
    // The label parses back exactly: decimal integers, shortest round-trip floats, TRUE/FALSE, ISO
    // dates, text and category labels.
    fill_constant(col, &levels[level])
}

/// `col` with its NaN cells made null (float columns; others unchanged), so
/// that the fills treat NaN as the missing value it is.
fn nan_to_null(col: &Column) -> MungeResult<Column> {
    let name = col.name().clone();
    Ok(match col.dtype() {
        DataType::Float64 => {
            // na-ok: NaN -> null conversion (the arm matched the dtype)
            let ca = col.f64()?;
            ca.apply(|v| v.filter(|x| !x.is_nan()))
                .with_name(name)
                .into_column()
        }
        DataType::Float32 => {
            // na-ok: NaN -> null conversion (the arm matched the dtype)
            let ca = col.f32()?;
            ca.apply(|v| v.filter(|x| !x.is_nan()))
                .with_name(name)
                .into_column()
        }
        _ => col.clone(),
    })
}

/// The column `name` of `df` (`col`, its NaN cells made null) with its missing
/// cells filled by the mean (`median` false) or the median of its observed
/// values, as Float64, and that value. A column without an observed value is an error: its
/// mean and median are undefined (R: `mean(c(NA, NA), na.rm = TRUE)` is NaN).
fn fill_with_statistic(
    df: &DataFrame,
    name: &str,
    col: &Column,
    median: bool,
) -> MungeResult<(Column, f64)> {
    let which = if median { "median" } else { "mean" };
    let cells = super::numeric_cells(df, name, false)?;
    let observed: Float64Chunked = cells.into_iter().flatten().map(Some).collect();
    let value = if median {
        observed.median()
    } else {
        observed.mean()
    }
    .ok_or_else(|| MungeError::InvalidValue {
        column: name.to_string(),
        value: which.to_string(),
        reason: format!(
            "the column has no observed values (all null or NaN), so its {which} is \
             undefined; drop the column or fill it with a constant"
        ),
    })?;
    let float_col = col.cast(&DataType::Float64)?;
    // na-ok: cast to Float64 above; the fill below replaces the missing cells
    let filled = float_col.f64()?.fill_null_with_values(value)?;
    Ok((filled.with_name(col.name().clone()).into_column(), value))
}

fn fill_forward(col: &Column) -> MungeResult<Column> {
    let filled = col.fill_null(FillNullStrategy::Forward(None))?;
    Ok(filled)
}

fn fill_backward(col: &Column) -> MungeResult<Column> {
    let filled = col.fill_null(FillNullStrategy::Backward(None))?;
    Ok(filled)
}

/// Remove duplicate rows.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `subset` - Columns to consider for duplicates (None = all columns)
/// * `keep` - Which duplicate to keep: "first", "last", or "none"
///
/// # Example
/// ```ignore
/// // Remove duplicates based on id column, keep first
/// let deduped = deduplicate(&dataset, Some(&["id"]), "first")?;
/// ```
pub fn deduplicate(dataset: &Dataset, subset: Option<&[&str]>, keep: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate columns if specified
    if let Some(cols) = subset {
        for col in cols {
            if df.column(col).is_err() {
                return Err(MungeError::ColumnNotFound(col.to_string()));
            }
        }
    }

    let keep_strategy = match keep.to_lowercase().as_str() {
        "first" => UniqueKeepStrategy::First,
        "last" => UniqueKeepStrategy::Last,
        "none" => UniqueKeepStrategy::None,
        "any" => UniqueKeepStrategy::Any,
        _ => {
            return Err(MungeError::InvalidExpression(format!(
                "Invalid 'keep' parameter: {}. Use 'first', 'last', 'none', or 'any'",
                keep
            )));
        }
    };

    let subset_cols: Option<Vec<String>> =
        subset.map(|cols| cols.iter().map(|&s| s.to_string()).collect());
    let unique_df = df.unique::<String, &str>(subset_cols.as_deref(), keep_strategy, None)?;

    Ok(Dataset::new(unique_df))
}

/// Cast a column to a different data type.
///
/// Values that cannot be converted become missing; [`cast_with`] reports
/// them, or refuses the cast with `strict`.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to cast
/// * `dtype` - Target type: "int", "float", "string", "bool"
///
/// # Example
/// ```ignore
/// let casted = cast(&dataset, "age", "float")?;
/// ```
pub fn cast(dataset: &Dataset, column: &str, dtype: &str) -> MungeResult<Dataset> {
    cast_with(dataset, column, dtype, &CastOptions::default()).map(|(ds, _)| ds)
}

/// Cast multiple columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `casts` - Vec of (column, dtype) pairs
pub fn cast_columns(dataset: &Dataset, casts: &[(&str, &str)]) -> MungeResult<Dataset> {
    cast_columns_with(dataset, casts, &CastOptions::default()).map(|(ds, _)| ds)
}

/// Options of [`cast_with`] and [`cast_columns_with`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CastOptions {
    /// Refuse the cast when any present value cannot be converted.
    pub strict: bool,
}

/// What a cast lost: the present values that became missing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CastReport {
    /// The column cast.
    pub column: String,
    /// Target type as requested ("int", "float", ...).
    pub dtype: String,
    /// Present (non-missing) input values that became missing.
    pub n_failed: usize,
    /// Up to 3 distinct failed inputs, in row order, as text.
    pub examples: Vec<String>,
}

impl CastReport {
    /// `column 'v': 2 values could not be converted to int and are now missing
    /// (first: '1,234', 'x')` ("1 value ... is now missing" in the singular);
    /// `None` when nothing failed.
    pub fn note(&self) -> Option<String> {
        (self.n_failed > 0).then(|| {
            let verb = if self.n_failed == 1 { "is" } else { "are" };
            format!(
                "column '{}': {} could not be converted to {} and {verb} now missing (first: {})",
                self.column,
                self.failed_values(),
                self.dtype,
                self.quoted_examples()
            )
        })
    }

    /// "1 value" / "2 values".
    fn failed_values(&self) -> String {
        match self.n_failed {
            1 => "1 value".to_string(),
            n => format!("{n} values"),
        }
    }

    /// The examples, each in single quotes, comma-separated.
    fn quoted_examples(&self) -> String {
        let quoted: Vec<String> = self.examples.iter().map(|e| format!("'{e}'")).collect();
        quoted.join(", ")
    }

    /// Remember a failed input (at most 3 distinct ones, in row order).
    fn failed(&mut self, value: impl FnOnce() -> String) {
        self.n_failed += 1;
        if self.examples.len() < 3 {
            let value = value();
            if !self.examples.contains(&value) {
                self.examples.push(value);
            }
        }
    }
}

/// Cast `column` to `dtype` ([`parse_cast_dtype`]'s names) and report the
/// present values that could not be converted and are now missing.
///
/// Text (and categorical labels) is converted with R's coercion rules:
/// surrounding whitespace is trimmed; `""` and `"NA"` are missing (not
/// failures, as R is silent about them); integer targets take a number and
/// truncate it toward zero (`as.integer("3.7")` is 3) and refuse one outside
/// the target's range; float targets parse a number (`inf` and `nan` too); the
/// boolean target accepts `TRUE`, `true`, `T`, `True` and `FALSE`, `false`,
/// `F`, `False`. Any other source is cast by polars (numbers truncate toward
/// zero, a non-finite or out-of-range number becomes missing), and a present
/// (neither null nor NaN) value that comes out missing is a failure.
///
/// # Errors
///
/// [`MungeError::ColumnNotFound`] for a missing column,
/// [`MungeError::InvalidExpression`] for an unknown type name,
/// [`MungeError::InvalidValue`] when `options.strict` and a value failed, and
/// the error of a cancelled or over-limit computation.
pub fn cast_with(
    dataset: &Dataset,
    column: &str,
    dtype: &str,
    options: &CastOptions,
) -> MungeResult<(Dataset, CastReport)> {
    let df = dataset.df();
    let col = df
        .column(column)
        .map_err(|_| MungeError::ColumnNotFound(column.to_string()))?;
    let target = parse_cast_dtype(dtype)?;
    let mut report = CastReport {
        column: column.to_string(),
        dtype: dtype.to_string(),
        ..CastReport::default()
    };

    let text_source = matches!(
        col.dtype(),
        DataType::String | DataType::Categorical(..) | DataType::Enum(..)
    );
    let casted = if text_source && target != DataType::String {
        let labels = col.cast(&DataType::String)?;
        cast_text(labels.str()?, &target, &mut report)?.with_name(col.name().clone())
    } else {
        let casted = col.cast(&target)?;
        count_lost_values(col, &casted, &mut report)?;
        casted
    };

    if options.strict && report.n_failed > 0 {
        return Err(MungeError::InvalidValue {
            column: column.to_string(),
            value: report.examples.first().cloned().unwrap_or_default(),
            reason: format!(
                "{} could not be converted to {dtype} (first: {}); fix them or cast without \
                 strict to set them missing",
                report.failed_values(),
                report.quoted_examples()
            ),
        });
    }

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(casted)?.clone();

    Ok((Dataset::new(result_df), report))
}

/// [`cast_with`] for each (column, dtype) pair in order, with one report per
/// pair.
pub fn cast_columns_with(
    dataset: &Dataset,
    casts: &[(&str, &str)],
    options: &CastOptions,
) -> MungeResult<(Dataset, Vec<CastReport>)> {
    let mut result = dataset.clone();
    let mut reports = Vec::with_capacity(casts.len());
    for (column, dtype) in casts {
        let (next, report) = cast_with(&result, column, dtype, options)?;
        result = next;
        reports.push(report);
    }
    Ok((result, reports))
}

/// A cooperative-cancellation checkpoint of a cast, as a munging error.
fn cast_checkpoint(row: usize) -> MungeResult<()> {
    if row.is_multiple_of(CHECKPOINT_EVERY) {
        crate::resource::checkpoint().map_err(|e| MungeError::FeatureError(e.to_string()))?;
    }
    Ok(())
}

/// Text values converted to `target` (not String) by R's coercion rules (see
/// [`cast_with`]); the failures go into `report`.
fn cast_text(
    text: &StringChunked,
    target: &DataType,
    report: &mut CastReport,
) -> MungeResult<Column> {
    let alloc = |e: crate::errors::EconError| MungeError::FeatureError(e.to_string());
    /// The trimmed text, or None for a missing value (null, "" or "NA").
    fn present(v: Option<&str>) -> Option<&str> {
        v.map(str::trim).filter(|t| !t.is_empty() && *t != "NA")
    }
    let name = text.name().clone();
    Ok(match target {
        DataType::Int64 | DataType::Int32 => {
            let int32 = *target == DataType::Int32;
            // [lo, hi) of the target; both bounds are exact in f64.
            let (lo, hi) = if int32 {
                (f64::from(i32::MIN), f64::from(i32::MAX) + 1.0)
            } else {
                (i64::MIN as f64, -(i64::MIN as f64))
            };
            let mut values =
                crate::resource::try_with_capacity::<Option<i64>>(text.len(), "cast values")
                    .map_err(alloc)?;
            for (row, v) in text.iter().enumerate() {
                cast_checkpoint(row)?;
                let value = present(v).and_then(|t| {
                    // An integer literal exactly, else a number truncated toward zero.
                    let exact = if int32 {
                        t.parse::<i32>().ok().map(i64::from)
                    } else {
                        t.parse::<i64>().ok()
                    };
                    let parsed = exact.or_else(|| {
                        t.parse::<f64>()
                            .ok()
                            .map(f64::trunc)
                            .filter(|x| x.is_finite() && *x >= lo && *x < hi)
                            .map(|x| x as i64)
                    });
                    if parsed.is_none() {
                        report.failed(|| t.to_string());
                    }
                    parsed
                });
                values.push(value);
            }
            if *target == DataType::Int32 {
                let ca: Int32Chunked = values
                    .into_iter()
                    .map(|v| v.and_then(|x| i32::try_from(x).ok()))
                    .collect();
                ca.with_name(name).into_column()
            } else {
                let ca: Int64Chunked = values.into_iter().collect();
                ca.with_name(name).into_column()
            }
        }
        DataType::Float64 | DataType::Float32 => {
            let mut values =
                crate::resource::try_with_capacity::<Option<f64>>(text.len(), "cast values")
                    .map_err(alloc)?;
            for (row, v) in text.iter().enumerate() {
                cast_checkpoint(row)?;
                let value = present(v).and_then(|t| {
                    let parsed = t.parse::<f64>().ok();
                    if parsed.is_none() {
                        report.failed(|| t.to_string());
                    }
                    parsed
                });
                values.push(value);
            }
            let ca: Float64Chunked = values.into_iter().collect();
            let col = ca.with_name(name).into_column();
            if *target == DataType::Float32 {
                col.cast(&DataType::Float32)?
            } else {
                col
            }
        }
        DataType::Boolean => {
            let mut values =
                crate::resource::try_with_capacity::<Option<bool>>(text.len(), "cast values")
                    .map_err(alloc)?;
            for (row, v) in text.iter().enumerate() {
                cast_checkpoint(row)?;
                let value = present(v).and_then(|t| {
                    let parsed = match t {
                        "TRUE" | "true" | "T" | "True" => Some(true),
                        "FALSE" | "false" | "F" | "False" => Some(false),
                        _ => None,
                    };
                    if parsed.is_none() {
                        report.failed(|| t.to_string());
                    }
                    parsed
                });
                values.push(value);
            }
            let ca: BooleanChunked = values.into_iter().collect();
            ca.with_name(name).into_column()
        }
        other => text.clone().into_column().cast(other)?,
    })
}

/// Count into `report` the present (neither null nor NaN) values of `source`
/// that are missing in `casted`, its polars cast.
fn count_lost_values(source: &Column, casted: &Column, report: &mut CastReport) -> MungeResult<()> {
    if casted.null_count() == source.null_count() {
        return Ok(());
    }
    let source = source.as_materialized_series();
    let casted = casted.as_materialized_series();
    let floats = source.dtype().is_float();
    for row in 0..source.len() {
        cast_checkpoint(row)?;
        let before = source.get(row)?;
        if before.is_null() || !casted.get(row)?.is_null() {
            continue;
        }
        let text = match before.extract::<f64>() {
            Some(x) if floats && x.is_nan() => continue, // NaN was missing already
            Some(x) if floats => x.to_string(),
            _ => before.to_string(),
        };
        report.failed(|| text);
    }
    Ok(())
}

/// The data type [`cast`] and [`cast_columns`] cast to for the type name
/// `dtype` (case-insensitive): `int`, `int64` or `integer` (Int64), `int32`,
/// `float`, `float64` or `double` (Float64), `float32`, `string`, `str` or
/// `text` (String), `bool` or `boolean` (Boolean).
///
/// # Errors
///
/// [`MungeError::InvalidExpression`] for any other name.
pub fn parse_cast_dtype(dtype: &str) -> MungeResult<DataType> {
    match dtype.to_lowercase().as_str() {
        "int" | "int64" | "integer" => Ok(DataType::Int64),
        "int32" => Ok(DataType::Int32),
        "float" | "float64" | "double" => Ok(DataType::Float64),
        "float32" => Ok(DataType::Float32),
        "string" | "str" | "text" => Ok(DataType::String),
        "bool" | "boolean" => Ok(DataType::Boolean),
        _ => Err(MungeError::InvalidExpression(format!(
            "Unknown dtype: {}. Use int, float, string, or bool",
            dtype
        ))),
    }
}

/// Clip numeric values to a range (R's `pmin(pmax(x, min), max)`).
///
/// Nulls stay null and NaN stays NaN; the result is Float64.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Numeric column to clip (text is refused, never cast to nulls)
/// * `min` - Minimum value (None = no lower bound)
/// * `max` - Maximum value (None = no upper bound)
pub fn clip(
    dataset: &Dataset,
    column: &str,
    min: Option<f64>,
    max: Option<f64>,
) -> MungeResult<Dataset> {
    let df = dataset.df();
    let ca = super::numeric_f64(df, column, false)?;

    let clipped: Float64Chunked = ca
        .iter()
        .map(|opt| {
            opt.map(|v| {
                // f64::max(NaN, min) is min: NaN must not become a bound.
                if v.is_nan() {
                    return v;
                }
                let mut val = v;
                if let Some(min_val) = min {
                    val = val.max(min_val);
                }
                if let Some(max_val) = max {
                    val = val.min(max_val);
                }
                val
            })
        })
        .collect();

    let clipped_col = clipped.with_name(column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(clipped_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Options of [`replace_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaceOptions {
    /// A new value of "" (after trim), "NA" (exactly) or "null" (any ASCII case) writes a missing
    /// value. Default true. `old_value` is always literal.
    pub null_tokens: bool,
}

impl Default for ReplaceOptions {
    fn default() -> Self {
        ReplaceOptions { null_tokens: true }
    }
}

/// What [`replace_with`] did.
#[derive(Clone)]
pub struct ReplaceOutcome {
    pub dataset: Dataset,
    pub column: String,
    pub old_value: String,
    /// `None`: the matches became missing.
    pub new_value: Option<String>,
    pub n_replaced: usize,
}

impl ReplaceOutcome {
    /// One sentence for the user: "Replaced 2 values '3' with missing in column 'k'", "Replaced 1
    /// value '-99' with '0' in column 'i32'" or "No value '-99' found in column 'k'; nothing was
    /// replaced".
    pub fn summary(&self) -> String {
        if self.n_replaced == 0 {
            return format!(
                "No value '{}' found in column '{}'; nothing was replaced",
                self.old_value, self.column
            );
        }
        let values = if self.n_replaced == 1 {
            "value"
        } else {
            "values"
        };
        let with = match &self.new_value {
            Some(v) => format!("'{v}'"),
            None => "missing".to_string(),
        };
        format!(
            "Replaced {} {values} '{}' with {with} in column '{}'",
            self.n_replaced, self.old_value, self.column
        )
    }
}

/// The new value of [`replace`] and [`replace_with`] that writes a missing value by default.
fn is_null_token(v: &str) -> bool {
    v.trim().is_empty() || v == "NA" || v.eq_ignore_ascii_case("null")
}

fn invalid_value(column: &str, value: &str, reason: String) -> MungeError {
    MungeError::InvalidValue {
        column: column.to_string(),
        value: value.to_string(),
        reason,
    }
}

/// An integer literal for a column of type `dtype`: decimal digits, or a finite integral number
/// such as "1e3" (as R's `as.integer`).
fn parse_int<T>(column: &str, dtype: &DataType, text: &str) -> MungeResult<T>
where
    T: TryFrom<i128> + std::str::FromStr,
{
    let t = text.trim();
    if let Ok(v) = t.parse::<T>() {
        return Ok(v);
    }
    // 2^127 as f64: below it the conversion to i128 is exact for an integral value.
    const I128_BOUND: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
    let wide = t.parse::<i128>().ok().or_else(|| {
        t.parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && v.fract() == 0.0 && v.abs() < I128_BOUND)
            .map(|v| v as i128)
    });
    let out_of_range = || format!("out of range for the column type {dtype}");
    match wide {
        Some(v) => T::try_from(v).map_err(|_| invalid_value(column, text, out_of_range())),
        // An integral number beyond i128 (or digits beyond it) is a number the type cannot hold.
        None if t
            .parse::<f64>()
            .is_ok_and(|v| v.is_finite() && v.fract() == 0.0) =>
        {
            Err(invalid_value(column, text, out_of_range()))
        }
        None => Err(invalid_value(
            column,
            text,
            format!("not an integer (the column is {dtype})"),
        )),
    }
}

fn parse_float(column: &str, dtype: &DataType, text: &str) -> MungeResult<f64> {
    text.trim().parse::<f64>().map_err(|_| {
        invalid_value(
            column,
            text,
            format!("not a number (the column is {dtype})"),
        )
    })
}

fn parse_f32(column: &str, dtype: &DataType, text: &str) -> MungeResult<f32> {
    let v = parse_float(column, dtype, text)?;
    if v.is_finite() && v.abs() > f64::from(f32::MAX) {
        return Err(invalid_value(
            column,
            text,
            format!("out of range for the column type {dtype}"),
        ));
    }
    Ok(v as f32)
}

/// A logical literal, as R's `as.logical`.
fn parse_bool(column: &str, text: &str) -> MungeResult<bool> {
    match text.trim() {
        "TRUE" | "true" | "T" | "True" => Ok(true),
        "FALSE" | "false" | "F" | "False" => Ok(false),
        _ => Err(invalid_value(
            column,
            text,
            "not a logical value (TRUE/FALSE, T/F, true/false; the column is bool)".to_string(),
        )),
    }
}

/// A `YYYY-MM-DD` literal as days since 1970-01-01, the physical value of a Date cell.
fn parse_date(column: &str, text: &str) -> MungeResult<i32> {
    let not_a_date = || {
        invalid_value(
            column,
            text,
            "not a date in YYYY-MM-DD form (the column is date)".to_string(),
        )
    };
    let date =
        chrono::NaiveDate::parse_from_str(text.trim(), "%Y-%m-%d").map_err(|_| not_a_date())?;
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).ok_or_else(not_a_date)?;
    i32::try_from(date.signed_duration_since(epoch).num_days()).map_err(|_| not_a_date())
}

/// Parses `old_value` with `parse`; a missing-value token that does not parse says that `replace`
/// cannot match a missing cell, instead of calling the token a bad literal.
fn parse_old<T>(
    column: &str,
    text: &str,
    parse: impl Fn(&str) -> MungeResult<T>,
) -> MungeResult<T> {
    parse(text).map_err(|e| {
        if is_null_token(text) {
            invalid_value(
                column,
                text,
                "a missing value cannot be matched by replace; use fill_na to fill missing values"
                    .to_string(),
            )
        } else {
            e
        }
    })
}

/// The cells of `ca` with every match of `is_match` set to `new`, and the number of matches.
fn replace_values<T>(
    ca: &ChunkedArray<T>,
    is_match: impl Fn(T::Native) -> bool,
    new: Option<T::Native>,
) -> (ChunkedArray<T>, usize)
where
    T: PolarsNumericType,
{
    let mut n = 0usize;
    let replaced: ChunkedArray<T> = ca
        .iter()
        .map(|cell| match cell {
            Some(v) if is_match(v) => {
                n += 1;
                new
            }
            other => other,
        })
        .collect();
    (replaced.with_name(ca.name().clone()), n)
}

// DATA-69 (Tier 4): the match keeps its absolute-epsilon rule.
fn float_matches(v: f64, old: f64) -> bool {
    (v - old).abs() < f64::EPSILON
}

/// The column with every cell equal to `old` replaced by `new` (`None`: missing), with its name and
/// type kept, and the number of cells replaced. `old` and `new` are literals of the column's type.
fn replace_column(col: &Column, old: &str, new: Option<&str>) -> MungeResult<(Column, usize)> {
    let dtype = col.dtype();
    let name = col.name().clone();
    let column = col.name().as_str();

    macro_rules! integer {
        ($method:ident, $native:ty) => {{
            let old_v = parse_old(column, old, |t| parse_int::<$native>(column, dtype, t))?;
            let new_v = new
                .map(|t| parse_int::<$native>(column, dtype, t))
                .transpose()?;
            let (ca, n) = replace_values(col.$method()?, |v| v == old_v, new_v);
            (ca.into_column(), n)
        }};
    }

    Ok(match dtype {
        DataType::Int8 => integer!(i8, i8),
        DataType::Int16 => integer!(i16, i16),
        DataType::Int32 => integer!(i32, i32),
        DataType::Int64 => integer!(i64, i64),
        DataType::Int128 => integer!(i128, i128),
        DataType::UInt8 => integer!(u8, u8),
        DataType::UInt16 => integer!(u16, u16),
        DataType::UInt32 => integer!(u32, u32),
        DataType::UInt64 => integer!(u64, u64),
        DataType::UInt128 => integer!(u128, u128),
        DataType::Float32 => {
            // A Float32 cell holds the f32 nearest to its literal: compare in f32, widened.
            let old_v = f64::from(parse_old(column, old, |t| parse_f32(column, dtype, t))?);
            let new_v = new.map(|t| parse_f32(column, dtype, t)).transpose()?;
            let (ca, n) = replace_values(col.f32()?, |v| float_matches(f64::from(v), old_v), new_v);
            (ca.into_column(), n)
        }
        DataType::Float64 => {
            let old_v = parse_old(column, old, |t| parse_float(column, dtype, t))?;
            let new_v = new.map(|t| parse_float(column, dtype, t)).transpose()?;
            // na-ok: dtype matched by the arm; values are compared, nulls stay null
            let (ca, n) = replace_values(col.f64()?, |v| float_matches(v, old_v), new_v);
            (ca.into_column(), n)
        }
        DataType::Boolean => {
            let old_v = parse_old(column, old, |t| parse_bool(column, t))?;
            let new_v = new.map(|t| parse_bool(column, t)).transpose()?;
            let mut n = 0usize;
            let replaced: BooleanChunked = col
                .bool()?
                .iter()
                .map(|cell| match cell {
                    Some(v) if v == old_v => {
                        n += 1;
                        new_v
                    }
                    other => other,
                })
                .collect();
            (replaced.with_name(name).into_column(), n)
        }
        DataType::String => {
            let (ca, n) = replace_text(col.str()?, old, new);
            (ca.with_name(name).into_column(), n)
        }
        DataType::Date => {
            let old_v = parse_old(column, old, |t| parse_date(column, t))?;
            let new_v = new.map(|t| parse_date(column, t)).transpose()?;
            let (days, n) = replace_values(col.date()?.physical(), |v| v == old_v, new_v);
            (days.into_date().into_column(), n)
        }
        DataType::Categorical(..) | DataType::Enum(..) => {
            let text = col.cast(&DataType::String)?;
            let (ca, n) = replace_text(text.str()?, old, new);
            let back = ca
                .with_name(name)
                .into_column()
                .strict_cast(dtype)
                .map_err(|_| {
                    invalid_value(
                        column,
                        new.unwrap_or(""),
                        "not a category of the enum column".to_string(),
                    )
                })?;
            (back, n)
        }
        DataType::Null => (col.clone(), 0),
        other => {
            return Err(MungeError::TypeMismatch {
                column: column.to_string(),
                expected: "a string, categorical, boolean, integer, float or date column (cast \
                           it first with munge_cast)"
                    .to_string(),
                found: other.to_string(),
            });
        }
    })
}

/// The cells of `ca` with every exact match of `old` replaced by `new`, and the number of matches.
fn replace_text(ca: &StringChunked, old: &str, new: Option<&str>) -> (StringChunked, usize) {
    let mut n = 0usize;
    let replaced: StringChunked = ca
        .iter()
        .map(|cell| match cell {
            Some(v) if v == old => {
                n += 1;
                new
            }
            other => other,
        })
        .collect();
    (replaced, n)
}

/// Replace the cells of a column equal to `old_value` by `new_value`, or by a missing value (R:
/// `dplyr::na_if(x, old)`, or `replace(x, x == old, new)`).
///
/// `new_value` is missing when it is `None` and, with [`ReplaceOptions::null_tokens`] (the
/// default), when it is "" (after trim), "NA" or "null" (any ASCII case). The values are parsed in
/// the column's own type, so the column keeps its type:
/// every integer width (with a range check), Float32 and Float64, Boolean (as R's `as.logical`),
/// Date (`YYYY-MM-DD`), String and Categorical or Enum labels (both compared exactly, so "NA" there
/// matches the text `NA`). A missing cell is never matched: filling one is `fill_na`'s job.
///
/// # Errors
///
/// [`MungeError::ColumnNotFound`]; [`MungeError::InvalidValue`] for a value the column's type cannot
/// hold; [`MungeError::TypeMismatch`] for any other column type (Datetime, Time, Duration, Decimal,
/// Binary, nested).
pub fn replace_with(
    dataset: &Dataset,
    column: &str,
    old_value: &str,
    new_value: Option<&str>,
    options: &ReplaceOptions,
) -> MungeResult<ReplaceOutcome> {
    let df = dataset.df();
    let col = df
        .column(column)
        .map_err(|_| MungeError::ColumnNotFound(column.to_string()))?;
    crate::resource::checkpoint().map_err(|e| invalid_value(column, old_value, e.to_string()))?;

    let new = new_value.filter(|v| !(options.null_tokens && is_null_token(v)));
    let (replaced, n_replaced) = replace_column(col, old_value, new)?;

    let mut result = df.clone();
    result.with_column(replaced)?;
    Ok(ReplaceOutcome {
        dataset: Dataset::new(result),
        column: column.to_string(),
        old_value: old_value.to_string(),
        new_value: new.map(str::to_string),
        n_replaced,
    })
}

/// Replace specific values in a column: [`replace_with`] with the default options, so a new value of
/// "", "NA" or "null" writes a missing value.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to modify
/// * `old_value` - Value to replace
/// * `new_value` - Replacement value
pub fn replace(
    dataset: &Dataset,
    column: &str,
    old_value: &str,
    new_value: &str,
) -> MungeResult<Dataset> {
    replace_with(
        dataset,
        column,
        old_value,
        Some(new_value),
        &ReplaceOptions::default(),
    )
    .map(|o| o.dataset)
}

/// Trim whitespace from string columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to trim (None = all string columns)
pub fn trim(dataset: &Dataset, columns: Option<&[&str]>) -> MungeResult<Dataset> {
    let df = dataset.df();

    let cols_to_trim: Vec<String> = match columns {
        Some(cols) => {
            for col in cols {
                if df.column(col).is_err() {
                    return Err(MungeError::ColumnNotFound(col.to_string()));
                }
            }
            cols.iter().map(|s| s.to_string()).collect()
        }
        None => df
            .get_column_names()
            .iter()
            .filter(|name| {
                df.column(name.as_str())
                    .map(|c| c.dtype() == &DataType::String)
                    .unwrap_or(false)
            })
            .map(|s| s.to_string())
            .collect(),
    };

    let mut result_df = df.clone();

    // Create a null pattern column for whitespace trimming
    let null_pattern = Column::full_null("pattern".into(), 1, &DataType::String);

    for col_name in &cols_to_trim {
        let col = result_df.column(col_name)?;
        if col.dtype() != &DataType::String {
            continue;
        }

        let ca = col.str()?;
        // Use vectorized strip_chars with null pattern (strips whitespace) - much faster
        let trimmed = ca.strip_chars(&null_pattern)?;

        let trimmed_col = trimmed.with_name(col.name().clone()).into_column();
        result_df = result_df.drop(col_name)?;
        result_df = result_df.with_column(trimmed_col)?.clone();
    }

    Ok(Dataset::new(result_df))
}

/// Convert string column to lowercase.
pub fn to_lowercase(dataset: &Dataset, column: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;
    // Use vectorized to_lowercase() - much faster than row-by-row iteration
    let lowered: StringChunked = ca.to_lowercase();

    let lowered_col = lowered.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(lowered_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Convert string column to uppercase.
pub fn to_uppercase(dataset: &Dataset, column: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;
    // Use vectorized to_uppercase() - much faster than row-by-row iteration
    let uppered: StringChunked = ca.to_uppercase();

    let uppered_col = uppered.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(uppered_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Replace substrings matching a regex pattern in a string column.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to modify
/// * `pattern` - Regex pattern to match
/// * `replacement` - Replacement string (can use capture groups like $1, $2)
///
/// # Example
/// ```ignore
/// // Remove all digits from a column
/// let cleaned = regex_replace(&dataset, "text", r"\d+", "")?;
///
/// // Standardize phone numbers
/// let cleaned = regex_replace(&dataset, "phone", r"(\d{3})(\d{3})(\d{4})", "($1) $2-$3")?;
/// ```
pub fn regex_replace(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    replacement: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let ca = col.str()?;
    let replaced: StringChunked = ca
        .into_iter()
        .map(|opt| opt.map(|s| re.replace_all(s, replacement).to_string()))
        .collect();

    let replaced_col = replaced.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(replaced_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// An upper bound of the text, in bytes, of the column [`regex_replace`]
/// would build from `column` with `pattern` and `replacement`, computed
/// without building it: each non-null value's bytes outside its matches,
/// and for each match the replacement's bytes plus, for each `$` in the
/// replacement (a capture-group reference expands to at most the match),
/// the match's bytes.
///
/// Validates the column and the pattern as [`regex_replace`] does and
/// allocates nothing per value, so a caller can check the size of the
/// output before replacing.
pub fn regex_replace_bytes(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    replacement: &str,
) -> MungeResult<u64> {
    let (ca, re) = split_input(dataset, column, pattern)?;
    let literal = replacement.len() as u64;
    let references = replacement.bytes().filter(|&b| b == b'$').count() as u64;
    let mut total = 0u64;
    for (row, opt) in ca.into_iter().enumerate() {
        if row.is_multiple_of(CHECKPOINT_EVERY) {
            split_checkpoint(column, pattern)?;
        }
        let Some(s) = opt else { continue };
        let mut bytes = s.len() as u64;
        for m in re.find_iter(s) {
            let matched = m.len() as u64;
            bytes = bytes
                .saturating_sub(matched)
                .saturating_add(literal)
                .saturating_add(references.saturating_mul(matched));
        }
        total = total.saturating_add(bytes);
    }
    Ok(total)
}

/// Replace all occurrences of a regex pattern (same as regex_replace but with clearer naming).
pub fn regex_replace_all(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    replacement: &str,
) -> MungeResult<Dataset> {
    regex_replace(dataset, column, pattern, replacement)
}

/// Extract substrings matching a regex pattern into a new column.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to extract from
/// * `pattern` - Regex pattern with capture groups
/// * `new_column` - Name for the new column containing extracted values
/// * `group` - Which capture group to extract (0 = entire match, 1 = first group, etc.)
///
/// # Example
/// ```ignore
/// // Extract email domain
/// let result = regex_extract(&dataset, "email", r"@([\w.-]+)", "domain", 1)?;
///
/// // Extract first number from text
/// let result = regex_extract(&dataset, "text", r"(\d+)", "first_number", 1)?;
/// ```
pub fn regex_extract(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    new_column: &str,
    group: usize,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let ca = col.str()?;
    let extracted: StringChunked = ca
        .into_iter()
        .map(|opt| {
            opt.and_then(|s| {
                re.captures(s)
                    .and_then(|caps| caps.get(group).map(|m| m.as_str().to_string()))
            })
        })
        .collect();

    let extracted_col = extracted.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(extracted_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Extract all matches of a regex pattern into a new column (as comma-separated string).
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to extract from
/// * `pattern` - Regex pattern to match
/// * `new_column` - Name for the new column containing all matches
/// * `separator` - Separator for multiple matches (default ",")
///
/// # Example
/// ```ignore
/// // Extract all numbers from text
/// let result = regex_extract_all(&dataset, "text", r"\d+", "numbers", Some(","))?;
/// ```
pub fn regex_extract_all(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    new_column: &str,
    separator: Option<&str>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let sep = separator.unwrap_or(",");

    let ca = col.str()?;
    let extracted: StringChunked = ca
        .into_iter()
        .map(|opt| {
            opt.map(|s| {
                let matches: Vec<&str> = re.find_iter(s).map(|m| m.as_str()).collect();
                if matches.is_empty() {
                    String::new()
                } else {
                    matches.join(sep)
                }
            })
        })
        .collect();

    let extracted_col = extracted.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(extracted_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Count the number of matches of a regex pattern in each row.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to search in
/// * `pattern` - Regex pattern to match
/// * `new_column` - Name for the new column containing counts
///
/// # Example
/// ```ignore
/// // Count digits in each string
/// let result = regex_count(&dataset, "text", r"\d", "digit_count")?;
/// ```
pub fn regex_count(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    new_column: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    let ca = col.str()?;
    let counts: Int64Chunked = ca
        .into_iter()
        .map(|opt| opt.map(|s| re.find_iter(s).count() as i64))
        .collect();

    let counts_col = counts.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(counts_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Split a string column by a pattern into multiple new columns.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to split
/// * `pattern` - Pattern to split on (regex or literal)
/// * `max_splits` - `Some(m)` splits at most `m` times (at most `m + 1`
///   columns); `None` splits on every match
/// * `prefix` - Prefix for new column names (columns named prefix_0, prefix_1, etc.)
///
/// # Example
/// ```ignore
/// // Split full name into parts
/// let result = str_split(&dataset, "full_name", r"\s+", Some(2), "name_part")?;
/// ```
pub fn str_split(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    max_splits: Option<usize>,
    prefix: &str,
) -> MungeResult<Dataset> {
    let df = dataset.df();
    let (ca, re) = split_input(dataset, column, pattern)?;
    let resource_err = |max_parts: usize, e: crate::errors::EconError| MungeError::InvalidValue {
        column: column.to_string(),
        value: format!("{max_parts} parts"),
        reason: e.to_string(),
    };

    // First pass: split every row (parts borrow from the column) and find how
    // many columns we need
    let mut max_parts = 0;
    let mut splits: Vec<Vec<&str>> = Vec::with_capacity(ca.len());
    for (row, opt) in ca.into_iter().enumerate() {
        if row.is_multiple_of(CHECKPOINT_EVERY) {
            split_checkpoint(column, pattern)?;
        }
        let parts = match opt {
            Some(s) => split_parts(&re, s, max_splits),
            None => Vec::new(),
        };
        max_parts = max_parts.max(parts.len());
        splits.push(parts);
    }

    // Cap max_parts if max_splits was specified
    if let Some(max) = max_splits {
        max_parts = max_parts.min(max.saturating_add(1));
    }

    // The rows x parts output, checked before it is built
    let cells = crate::resource::checked_len(&[df.height(), max_parts], "str_split output")
        .map_err(|e| resource_err(max_parts, e))?;
    crate::resource::check_alloc::<[u8; 16]>(cells, "str_split output")
        .map_err(|e| resource_err(max_parts, e))?;

    // Create columns for each split part
    let mut result_df = df.clone();
    for i in 0..max_parts {
        split_checkpoint(column, pattern)?;
        let col_name = format!("{}_{}", prefix, i);
        let values: StringChunked = splits.iter().map(|parts| parts.get(i).copied()).collect();
        let new_col = values.with_name(col_name.into()).into_column();
        result_df = result_df.with_column(new_col)?.clone();
    }

    Ok(Dataset::new(result_df))
}

/// The string column `column` and the compiled `pattern`: the input checks
/// [`str_split`] and [`str_split_part_count`] share.
fn split_input<'a>(
    dataset: &'a Dataset,
    column: &str,
    pattern: &str,
) -> MungeResult<(&'a StringChunked, Regex)> {
    let col = dataset.df().column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let re = Regex::new(pattern).map_err(|e| {
        MungeError::InvalidExpression(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    Ok((col.str()?, re))
}

/// A cooperative-cancellation checkpoint in [`str_split`] or
/// [`str_split_part_count`]: both report a stop as the same error.
fn split_checkpoint(column: &str, pattern: &str) -> MungeResult<()> {
    crate::resource::checkpoint().map_err(|e| MungeError::InvalidValue {
        column: column.to_string(),
        value: pattern.to_string(),
        reason: e.to_string(),
    })
}

/// The parts of `s` split on `re`: at most `max_splits + 1` parts when
/// `max_splits` is given (saturating), otherwise split on every match.
fn split_parts<'s>(re: &Regex, s: &'s str, max_splits: Option<usize>) -> Vec<&'s str> {
    match max_splits {
        Some(m) => re.splitn(s, m.saturating_add(1)).collect(),
        None => re.split(s).collect(),
    }
}

/// The number of parts [`split_parts`] would return, without allocating.
fn count_parts(re: &Regex, s: &str, max_splits: Option<usize>) -> usize {
    match max_splits {
        Some(m) => re.splitn(s, m.saturating_add(1)).count(),
        None => re.split(s).count(),
    }
}

/// The largest number of parts any row of `column` splits into under
/// `pattern` and `max_splits`: the number of columns [`str_split`] would add
/// (at most `max_splits + 1`, saturating; `0` when every row is null).
///
/// Validates the column and the pattern as [`str_split`] does and allocates
/// nothing per part, so a caller can check the output width before splitting.
pub fn str_split_part_count(
    dataset: &Dataset,
    column: &str,
    pattern: &str,
    max_splits: Option<usize>,
) -> MungeResult<usize> {
    let (ca, re) = split_input(dataset, column, pattern)?;
    let mut max_parts = 0;
    for (row, opt) in ca.into_iter().enumerate() {
        if row.is_multiple_of(CHECKPOINT_EVERY) {
            split_checkpoint(column, pattern)?;
        }
        if let Some(s) = opt {
            max_parts = max_parts.max(count_parts(&re, s, max_splits));
        }
    }
    Ok(max_parts)
}

/// Concatenate multiple string columns into one.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `columns` - Columns to concatenate
/// * `new_column` - Name for the concatenated column
/// * `separator` - Separator between values (default "")
///
/// # Example
/// ```ignore
/// let result = str_concat(&dataset, &["first_name", "last_name"], "full_name", Some(" "))?;
/// ```
pub fn str_concat(
    dataset: &Dataset,
    columns: &[&str],
    new_column: &str,
    separator: Option<&str>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    // Validate all columns exist and are strings
    for col_name in columns {
        let col = df.column(col_name)?;
        if col.dtype() != &DataType::String {
            return Err(MungeError::TypeMismatch {
                column: col_name.to_string(),
                expected: "String".to_string(),
                found: format!("{:?}", col.dtype()),
            });
        }
    }

    let sep = separator.unwrap_or("");

    // Collect string chunked arrays for all columns
    let string_cols: Vec<&StringChunked> = columns
        .iter()
        .map(|name| df.column(name).unwrap().str().unwrap())
        .collect();

    // Use vectorized hor_str_concat - much faster than row-by-row iteration
    let concatenated = hor_str_concat(&string_cols, sep, true)?;

    let concat_col = concatenated.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(concat_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Pad a string column to a fixed width.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to pad
/// * `width` - Target width
/// * `pad_char` - Character to pad with (default space)
/// * `side` - "left" or "right" (default "left")
pub fn str_pad(
    dataset: &Dataset,
    column: &str,
    width: usize,
    pad_char: Option<char>,
    side: Option<&str>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let pad = pad_char.unwrap_or(' ');
    let pad_side = side.unwrap_or("left");
    let pad_len = pad.len_utf8();
    let size_err = |reason: String| MungeError::InvalidValue {
        column: column.to_string(),
        value: format!("width {width}"),
        reason,
    };

    let ca = col.str()?;

    // Upper bound of the padding bytes, checked before anything is built
    let non_null = ca.len() - ca.null_count();
    let bound = crate::resource::checked_len(&[non_null, width, pad_len], "str_pad output")
        .map_err(|e| size_err(e.to_string()))?;
    crate::resource::check_alloc::<u8>(bound, "str_pad output")
        .map_err(|e| size_err(e.to_string()))?;

    let mut values: Vec<Option<String>> = Vec::with_capacity(ca.len());
    for (row, opt) in ca.into_iter().enumerate() {
        if row.is_multiple_of(CHECKPOINT_EVERY) {
            crate::resource::checkpoint().map_err(|e| size_err(e.to_string()))?;
        }
        let Some(s) = opt else {
            values.push(None);
            continue;
        };
        let len = s.chars().count();
        if len >= width {
            values.push(Some(s.to_string()));
            continue;
        }
        let n_pad = width - len;
        let total = n_pad
            .checked_mul(pad_len)
            .and_then(|b| b.checked_add(s.len()))
            .ok_or_else(|| size_err("the padded string length overflows".to_string()))?;
        let mut padded = String::new();
        padded.try_reserve_exact(total).map_err(|_| {
            size_err(format!(
                "a padded value would need {total} bytes, which could not be allocated"
            ))
        })?;
        let left = matches!(pad_side, "left" | "start");
        if !left {
            padded.push_str(s);
        }
        padded.extend(std::iter::repeat_n(pad, n_pad));
        if left {
            padded.push_str(s);
        }
        values.push(Some(padded));
    }
    let padded: StringChunked = values.into_iter().collect();

    let padded_col = padded.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(padded_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Get substring from a string column.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to extract from
/// * `start` - Start index (0-based, supports negative for from-end)
/// * `length` - Optional length to extract
pub fn str_substring(
    dataset: &Dataset,
    column: &str,
    start: i64,
    length: Option<usize>,
) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;

    // Use vectorized str_slice() - much faster than row-by-row iteration
    // Create scalar columns for offset and length
    let offset_col = Column::new_scalar("offset".into(), Scalar::from(start), 1);
    let length_col = match length {
        Some(l) => Column::new_scalar("length".into(), Scalar::from(l as u64), 1),
        None => Column::full_null("length".into(), 1, &DataType::UInt64),
    };

    let substr = ca.str_slice(&offset_col, &length_col)?;

    let substr_col = substr.with_name(col.name().clone()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.drop(column)?;
    result_df = result_df.with_column(substr_col)?.clone();

    Ok(Dataset::new(result_df))
}

/// Get the length of strings in a column (counts Unicode characters).
///
/// Uses SIMD-optimized `bytecount::num_chars` for fast UTF-8 character counting.
///
/// # Arguments
/// * `dataset` - Source dataset
/// * `column` - Column to measure
/// * `new_column` - Name for the length column
pub fn str_length(dataset: &Dataset, column: &str, new_column: &str) -> MungeResult<Dataset> {
    let df = dataset.df();

    let col = df.column(column)?;
    if col.dtype() != &DataType::String {
        return Err(MungeError::TypeMismatch {
            column: column.to_string(),
            expected: "String".to_string(),
            found: format!("{:?}", col.dtype()),
        });
    }

    let ca = col.str()?;
    // Use SIMD-optimized bytecount::num_chars for fast UTF-8 character counting
    // This is significantly faster than std chars().count() or Polars str_len_chars()
    let lengths: Int64Chunked = ca
        .into_iter()
        .map(|opt| opt.map(|s| bytecount::num_chars(s.as_bytes()) as i64))
        .collect();

    let length_col = lengths.with_name(new_column.into()).into_column();

    let mut result_df = df.clone();
    result_df = result_df.with_column(length_col)?.clone();

    Ok(Dataset::new(result_df))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    #[test]
    fn regex_replace_bytes_bounds_the_replaced_text() {
        let df = df! {
            "s" => [Some("ab12cd345"), None, Some(""), Some("xyz"), Some("9")],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let text = |out: &Dataset| -> u64 {
            out.df()
                .column("s")
                .unwrap()
                .str()
                .unwrap()
                .into_iter()
                .flatten()
                .map(|s| s.len() as u64)
                .sum()
        };
        // Literal replacements, group references, `$$`, empty matches.
        for (pattern, replacement) in [
            (r"\d+", "#"),
            (r"\d", "<$0$0>"),
            (r"(\d)(\d)?", "${2}-$1-$$"),
            ("", "_"),
            ("z*", "ZZ"),
            (r"[a-z]", ""),
        ] {
            let bound = regex_replace_bytes(&ds, "s", pattern, replacement).unwrap();
            let built = text(&regex_replace(&ds, "s", pattern, replacement).unwrap());
            assert!(
                bound >= built,
                "{pattern:?} -> {replacement:?}: bound {bound} < {built}"
            );
        }
        // A literal replacement is exact: "ab#cd#", "", "xyz" and "#".
        assert_eq!(regex_replace_bytes(&ds, "s", r"\d+", "#").unwrap(), 10);
        // The input checks are regex_replace's.
        assert!(matches!(
            regex_replace_bytes(&ds, "s", "(", "x"),
            Err(MungeError::InvalidExpression(_))
        ));
        assert!(regex_replace_bytes(&ds, "nope", "a", "b").is_err());
    }

    fn split_col(ds: &Dataset, name: &str) -> Vec<Option<String>> {
        ds.df()
            .column(name)
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|v| v.map(str::to_string))
            .collect()
    }

    /// The error text of a call that must fail (`Dataset` is not `Debug`).
    fn err_text(r: MungeResult<Dataset>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn str_split_default_path_splits_on_every_match() {
        let ds = Dataset::new(df!("s" => [Some("a,b,c"), Some("d"), None]).unwrap());
        let out = str_split(&ds, "s", ",", None, "p").unwrap();
        assert_eq!(
            split_col(&out, "p_0"),
            vec![Some("a".into()), Some("d".into()), None]
        );
        assert_eq!(split_col(&out, "p_1"), vec![Some("b".into()), None, None]);
        assert_eq!(split_col(&out, "p_2"), vec![Some("c".into()), None, None]);
        assert!(out.df().column("p_3").is_err());
        assert!(
            str_split(&ds, "s", ",", Some(usize::MAX), "p")
                .unwrap()
                .df()
                .column("p_2")
                .is_ok()
        );
        let one = str_split(&ds, "s", ",", Some(1), "p").unwrap();
        assert_eq!(split_col(&one, "p_1"), vec![Some("b,c".into()), None, None]);
        assert!(one.df().column("p_2").is_err());
    }

    #[test]
    fn str_split_part_count_and_output_size() {
        let ds = Dataset::new(df!("s" => [Some("a,b,c"), Some("d"), None]).unwrap());
        assert_eq!(str_split_part_count(&ds, "s", ",", None).unwrap(), 3);
        assert_eq!(str_split_part_count(&ds, "s", ",", Some(0)).unwrap(), 1);
        assert_eq!(str_split_part_count(&ds, "s", ",", Some(1)).unwrap(), 2);
        assert_eq!(
            str_split_part_count(&ds, "s", ",", Some(usize::MAX)).unwrap(),
            3
        );
        let rows = Dataset::new(df!("s" => vec!["a,b,c"; 10]).unwrap());
        let r =
            crate::resource::with_alloc_ceiling(Some(64), || str_split(&rows, "s", ",", None, "p"));
        assert!(err_text(r).contains("limit")); // 10 x 3 cells x 16 bytes
    }

    #[test]
    fn str_split_and_part_count_report_a_stop_alike() {
        let ds = Dataset::new(df!("s" => ["a,b", "c"]).unwrap());
        let token = crate::resource::CancelToken::new();
        token.cancel();
        let (split, count) =
            crate::resource::with_budget(crate::resource::Budget::new(None, token), || {
                (
                    err_text(str_split(&ds, "s", ",", None, "p")),
                    str_split_part_count(&ds, "s", ",", None)
                        .unwrap_err()
                        .to_string(),
                )
            });
        assert!(split.contains("cancelled"), "{split}");
        assert_eq!(split, count);
    }

    #[test]
    fn str_pad_impossible_width_is_an_error() {
        let ds = Dataset::new(df!("s" => ["ab", "abc"]).unwrap());
        assert!(str_pad(&ds, "s", usize::MAX / 2, Some('x'), None).is_err());
        let ok = str_pad(&ds, "s", 5, Some('0'), Some("left")).unwrap();
        assert_eq!(
            split_col(&ok, "s"),
            vec![Some("000ab".into()), Some("00abc".into())]
        );
    }

    fn test_dataset_with_nulls() -> Dataset {
        let df = df! {
            "id" => [Some(1), Some(2), Some(3), Some(4), Some(5)],
            "name" => [Some("Alice"), None, Some("Charlie"), Some("Diana"), None],
            "age" => [Some(25), Some(30), None, Some(28), Some(22)],
            "score" => [Some(85.5), None, Some(78.5), None, Some(95.5)],
        }
        .unwrap();
        Dataset::new(df)
    }

    fn test_dataset_with_duplicates() -> Dataset {
        let df = df! {
            "id" => [1, 1, 2, 2, 3],
            "name" => ["Alice", "Alice", "Bob", "Bob", "Charlie"],
            "value" => [10, 10, 20, 25, 30],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_drop_na_any() {
        let ds = test_dataset_with_nulls();
        let clean = drop_na(&ds, None, "any").unwrap();
        assert_eq!(clean.nrows(), 1); // Only row 1 (id=1) has no nulls in any column
    }

    #[test]
    fn test_drop_na_specific_columns() {
        let ds = test_dataset_with_nulls();
        let clean = drop_na(&ds, Some(&["age"]), "any").unwrap();
        assert_eq!(clean.nrows(), 4); // Only row 3 has null age
    }

    #[test]
    fn test_fill_na_constant() {
        let ds = test_dataset_with_nulls();
        let filled = fill_na(&ds, Some(&["age"]), FillStrategy::Constant("0".to_string())).unwrap();
        let null_count = filled.df().column("age").unwrap().null_count();
        assert_eq!(null_count, 0);
    }

    #[test]
    fn test_fill_na_mean() {
        let ds = test_dataset_with_nulls();
        let filled = fill_na(&ds, Some(&["score"]), FillStrategy::Mean).unwrap();
        let null_count = filled.df().column("score").unwrap().null_count();
        assert_eq!(null_count, 0);
    }

    #[test]
    fn test_fill_na_forward() {
        let ds = test_dataset_with_nulls();
        let filled = fill_na(&ds, Some(&["name"]), FillStrategy::Forward).unwrap();
        // Row 2's name should be filled with "Alice" (forward fill)
        let names: Vec<Option<&str>> = filled
            .df()
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(names[1], Some("Alice"));
    }

    #[test]
    fn test_deduplicate_first() {
        let ds = test_dataset_with_duplicates();
        let deduped = deduplicate(&ds, Some(&["id"]), "first").unwrap();
        assert_eq!(deduped.nrows(), 3);
    }

    #[test]
    fn test_deduplicate_all_columns() {
        let ds = test_dataset_with_duplicates();
        let deduped = deduplicate(&ds, None, "first").unwrap();
        assert_eq!(deduped.nrows(), 4); // Two identical rows removed
    }

    #[test]
    fn test_cast() {
        let df = df! {
            "value" => ["1", "2", "3"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let casted = cast(&ds, "value", "int").unwrap();
        assert_eq!(
            casted.df().column("value").unwrap().dtype(),
            &DataType::Int64
        );
    }

    #[test]
    fn test_clip() {
        let df = df! {
            "value" => [1.0, 5.0, 10.0, 15.0, 20.0],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let clipped = clip(&ds, "value", Some(5.0), Some(15.0)).unwrap();
        let values: Vec<f64> = clipped
            .df()
            .column("value")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![5.0, 5.0, 10.0, 15.0, 15.0]);
    }

    #[test]
    fn test_replace() {
        let df = df! {
            "status" => ["active", "inactive", "active", "pending"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let replaced = replace(&ds, "status", "inactive", "disabled").unwrap();
        let statuses: Vec<&str> = replaced
            .df()
            .column("status")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!(statuses.contains(&"disabled"));
        assert!(!statuses.contains(&"inactive"));
    }

    #[test]
    fn test_trim() {
        let df = df! {
            "name" => ["  Alice  ", "Bob", "  Charlie"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let trimmed = trim(&ds, Some(&["name"])).unwrap();
        let names: Vec<&str> = trimmed
            .df()
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie"]);
    }

    #[test]
    fn test_to_lowercase() {
        let df = df! {
            "name" => ["ALICE", "Bob", "CHARLIE"],
        }
        .unwrap();
        let ds = Dataset::new(df);
        let lowered = to_lowercase(&ds, "name").unwrap();
        let names: Vec<&str> = lowered
            .df()
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(names, vec!["alice", "bob", "charlie"]);
    }

    /// A numeric column as `Option<f64>` (cast to Float64 first).
    fn f64s(ds: &Dataset, c: &str) -> Vec<Option<f64>> {
        ds.df()
            .column(c)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .collect()
    }

    #[test]
    fn test_fill_na_mean_of_all_missing_column_is_an_error() {
        // DATA-35. R: mean(c(NA_real_, NA_real_), na.rm = TRUE) is NaN and the median NA:
        // there is no value to fill with, and 0 is not one.
        let ds = Dataset::new(df!("id" => [1, 2, 3], "z" => [None::<f64>, None, None]).unwrap());
        for s in [FillStrategy::Mean, FillStrategy::Median] {
            let e = err_text(fill_na(&ds, Some(&["z"]), s));
            assert!(e.contains("'z'") && e.contains("no observed values"), "{e}");
        }
        // Also when no column is named.
        let ints = Dataset::new(df!("k" => [None::<i64>, None]).unwrap());
        assert!(fill_na(&ints, None, FillStrategy::Mean).is_err());
        // An all-empty CSV column is text: named, it has no observed values either.
        let text = Dataset::new(df!("t" => [None::<&str>, None]).unwrap());
        let e = err_text(fill_na(&text, Some(&["t"]), FillStrategy::Mean));
        assert!(e.contains("no observed values"), "{e}");
    }

    #[test]
    fn test_fill_na_uses_present_values_and_fills_nan() {
        // R: x <- c(1, NA, NaN, 4, 10); mean(x, na.rm = TRUE) = 5, median 4; is.na(NaN) is
        // TRUE, so tidyr::replace_na fills the NaN as well.
        let ds = Dataset::new(
            df!("x" => [Some(1.0), None, Some(f64::NAN), Some(4.0), Some(10.0)]).unwrap(),
        );
        let run = |s| f64s(&fill_na(&ds, Some(&["x"]), s).unwrap(), "x");
        assert_eq!(
            run(FillStrategy::Mean),
            vec![Some(1.0), Some(5.0), Some(5.0), Some(4.0), Some(10.0)]
        );
        assert_eq!(
            run(FillStrategy::Median),
            vec![Some(1.0), Some(4.0), Some(4.0), Some(4.0), Some(10.0)]
        );
        assert_eq!(
            run(FillStrategy::Zero),
            vec![Some(1.0), Some(0.0), Some(0.0), Some(4.0), Some(10.0)]
        );
        assert_eq!(
            run(FillStrategy::Forward),
            vec![Some(1.0), Some(1.0), Some(1.0), Some(4.0), Some(10.0)]
        );
        assert_eq!(
            run(FillStrategy::Constant("-1".into())),
            vec![Some(1.0), Some(-1.0), Some(-1.0), Some(4.0), Some(10.0)]
        );
    }

    #[test]
    fn test_na_fill_mean_fixtures() {
        // Int64 like Float64 (R: mean(c(1, 4)) = 2.5); a named text column is an
        // error for mean; an absent column is named; with no column named, text columns are
        // skipped as before.
        let i = fill_na(
            &Dataset::new(df!("v" => [Some(1i64), None, Some(4)]).unwrap()),
            Some(&["v"]),
            FillStrategy::Mean,
        )
        .unwrap();
        assert_eq!(f64s(&i, "v"), vec![Some(1.0), Some(2.5), Some(4.0)]);
        let ds = test_dataset_with_nulls();
        let e = err_text(fill_na(&ds, Some(&["name"]), FillStrategy::Mean));
        assert!(e.contains("'name'") && e.contains("numeric"), "{e}");
        assert!(matches!(
            fill_na(&ds, Some(&["nope"]), FillStrategy::Mean),
            Err(MungeError::ColumnNotFound(_))
        ));
        assert!(fill_na(&ds, None, FillStrategy::Mean).is_ok());
    }

    #[test]
    fn test_cast_text_to_number_follows_r_coercion() {
        // DATA-37. R:
        // v <- c("1", "1,234", "N/A", "3.7", "42", " 7", "$5", "", "-0.5e2")
        // as.integer(v)   # 1 NA NA 3 42 7 NA NA -50 (warning; "" is NA silently)
        // as.numeric(v)   # 1 NA NA 3.7 42 7 NA NA -50
        let ds = Dataset::new(
            df!("v" => ["1", "1,234", "N/A", "3.7", "42", " 7", "$5", "", "-0.5e2"]).unwrap(),
        );
        let ints = cast(&ds, "v", "int").unwrap();
        assert_eq!(
            ints.df()
                .column("v")
                .unwrap()
                .i64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![
                Some(1),
                None,
                None,
                Some(3),
                Some(42),
                Some(7),
                None,
                None,
                Some(-50)
            ]
        );
        assert_eq!(
            f64s(&cast(&ds, "v", "float").unwrap(), "v"),
            vec![
                Some(1.0),
                None,
                None,
                Some(3.7),
                Some(42.0),
                Some(7.0),
                None,
                None,
                Some(-50.0)
            ]
        );
        let (_, rep) = cast_with(&ds, "v", "int", &CastOptions::default()).unwrap();
        assert_eq!(
            (rep.n_failed, rep.examples.clone()),
            (3, vec!["1,234".into(), "N/A".into(), "$5".into()])
        );
        assert_eq!((rep.column.as_str(), rep.dtype.as_str()), ("v", "int"));
    }

    #[test]
    fn test_cast_reports_values_that_fail_to_convert() {
        // R: as.integer(c("1,234", "7", "x")) is NA 7 NA, "NAs introduced by coercion".
        let ds = Dataset::new(df!("v" => ["1,234", "7", "x"]).unwrap());
        let (out, rep) = cast_with(&ds, "v", "int", &CastOptions::default()).unwrap();
        assert_eq!(
            out.df()
                .column("v")
                .unwrap()
                .i64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![None, Some(7), None]
        );
        assert_eq!(
            rep.note().unwrap(),
            "column 'v': 2 values could not be converted to int and are now missing \
             (first: '1,234', 'x')"
        );
        let e = cast_with(&ds, "v", "int", &CastOptions { strict: true })
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("2 values could not be converted to int"), "{e}");
        let one = Dataset::new(df!("v" => ["7", "x"]).unwrap());
        let (_, rep) = cast_with(&one, "v", "int", &CastOptions::default()).unwrap();
        assert_eq!(
            rep.note().unwrap(),
            "column 'v': 1 value could not be converted to int and is now missing (first: 'x')"
        );
        let (_, ok) = cast_with(
            &Dataset::new(df!("v" => [Some("1"), Some(" 2 "), Some(""), None]).unwrap()),
            "v",
            "float",
            &CastOptions::default(),
        )
        .unwrap();
        assert!(ok.n_failed == 0 && ok.note().is_none(), "{ok:?}");
    }

    #[test]
    fn test_cast_text_to_bool_and_float_to_int() {
        // R: as.logical(c("TRUE", "true", "T", "True", "FALSE", "false", "F", "False", "yes", "1"))
        // T T T T F F F F NA NA; as.integer(c(3.7, -3.7, NaN, Inf)) is 3 -3 NA NA (a warning
        // for Inf only: NaN was already missing).
        let b = Dataset::new(
            df!("v" => ["TRUE", "true", "T", "True", "FALSE", "false", "F", "False", "yes", "1"])
                .unwrap(),
        );
        let (out, rep) = cast_with(&b, "v", "bool", &CastOptions::default()).unwrap();
        let (t, f) = (Some(true), Some(false));
        assert_eq!(
            out.df()
                .column("v")
                .unwrap()
                .bool()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![t, t, t, t, f, f, f, f, None, None]
        );
        assert_eq!(rep.n_failed, 2);
        let x = Dataset::new(df!("v" => [3.7, -3.7, f64::NAN, f64::INFINITY]).unwrap());
        let (out, rep) = cast_with(&x, "v", "int", &CastOptions::default()).unwrap();
        assert_eq!(
            out.df()
                .column("v")
                .unwrap()
                .i64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![Some(3), Some(-3), None, None]
        );
        assert_eq!(
            (rep.n_failed, rep.examples.clone()),
            (1, vec!["inf".to_string()])
        );
    }

    #[test]
    fn test_cast_columns_with_reports_each_column() {
        let ds = Dataset::new(df!("a" => ["1", "x"], "b" => ["2.5", "3"]).unwrap());
        let (out, reports) = cast_columns_with(
            &ds,
            &[("a", "int"), ("b", "float")],
            &CastOptions::default(),
        )
        .unwrap();
        assert_eq!(f64s(&out, "b"), vec![Some(2.5), Some(3.0)]);
        assert_eq!(
            reports.iter().map(|r| r.n_failed).collect::<Vec<_>>(),
            vec![1, 0]
        );
        assert!(matches!(
            cast_with(&ds, "nope", "int", &CastOptions::default()),
            Err(MungeError::ColumnNotFound(_))
        ));
    }

    #[test]
    fn test_clip_keeps_nan_and_refuses_text() {
        // R: pmin(pmax(c(NaN, -1, 5, NA, 2), 0), 3) is NaN 0 3 NA 2.
        let ds = Dataset::new(
            df!("v" => [Some(f64::NAN), Some(-1.0), Some(5.0), None, Some(2.0)]).unwrap(),
        );
        let got = f64s(&clip(&ds, "v", Some(0.0), Some(3.0)).unwrap(), "v");
        assert!(got[0].unwrap().is_nan());
        assert_eq!(&got[1..], &[Some(0.0), Some(3.0), None, Some(2.0)]);
        let e = err_text(clip(
            &Dataset::new(df!("v" => ["1", "5"]).unwrap()),
            "v",
            Some(0.0),
            None,
        ));
        assert!(e.contains("'v'") && e.contains("numeric"), "{e}");
        // Int64 accepted.
        let i = clip(
            &Dataset::new(df!("v" => [1i64, 9]).unwrap()),
            "v",
            None,
            Some(5.0),
        )
        .unwrap();
        assert_eq!(f64s(&i, "v"), vec![Some(1.0), Some(5.0)]);
    }

    /// Cells of any numeric/boolean column as f64 (null -> None).
    fn num_cells(ds: &Dataset, c: &str) -> Vec<Option<f64>> {
        f64s(ds, c)
    }
    /// Cells as text (categoricals by label; null -> None).
    fn text_cells(ds: &Dataset, c: &str) -> Vec<Option<String>> {
        ds.df()
            .column(c)
            .unwrap()
            .cast(&DataType::String)
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|v| v.map(str::to_string))
            .collect()
    }
    /// R's NA as NaN, to compare with a case vector (`Check` treats NaN as equal to NaN). Every test that
    /// uses it also asserts the null count, because NaN would pass the comparison.
    fn as_r(v: &[Option<f64>]) -> Vec<f64> {
        v.iter().map(|x| x.unwrap_or(f64::NAN)).collect()
    }
    /// Text cells against a `.strings()` case value (missing -> "").
    fn as_r_names(v: &[Option<String>]) -> Vec<String> {
        v.iter().map(|x| x.clone().unwrap_or_default()).collect()
    }
    fn dtype(ds: &Dataset, c: &str) -> DataType {
        ds.df().column(c).unwrap().dtype().clone()
    }
    /// The error text of any munging result that must fail (outcomes holding a Dataset are not Debug).
    fn err_any<T>(r: MungeResult<T>) -> String {
        match r {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn test_replace_with_none_gives_null() {
        // DATA-9 repro: "null" was written as the text 'null' (String) or refused ("Cannot parse as
        // integer/float"); Int32 came back unchanged. R: dplyr::na_if(x, sentinel).
        let ds = Dataset::new(
            df!("s" => ["ok", "N/A", "ok"], "i" => [1i64, -99, 3],
                                  "f" => [1.5, -99.0, 3.5], "i32" => [1i32, -99, 3])
            .unwrap(),
        );
        let s = replace(&ds, "s", "N/A", "null").unwrap();
        assert_eq!(
            text_cells(&s, "s"),
            [Some("ok".to_string()), None, Some("ok".to_string())]
        );
        for (c, token) in [("i", "null"), ("f", "NA"), ("i32", "")] {
            let out = replace(&ds, c, "-99", token).unwrap();
            assert_eq!(
                out.df().column(c).unwrap().null_count(),
                1,
                "{c} with {token:?}"
            );
            assert_eq!(dtype(&out, c), dtype(&ds, c), "{c}: the type is kept");
            assert_eq!(num_cells(&out, c)[2], num_cells(&ds, c)[2]);
        }
        // None is missing whatever the options; with the tokens off, "NA" is literal text in a text
        // column and an error in an integer column
        let off = ReplaceOptions { null_tokens: false };
        let o = replace_with(&ds, "i", "-99", None, &off).unwrap();
        assert_eq!(
            (
                o.n_replaced,
                o.dataset.df().column("i").unwrap().null_count()
            ),
            (1, 1)
        );
        assert_eq!(
            o.summary(),
            "Replaced 1 value '-99' with missing in column 'i'"
        );
        let lit = replace_with(&ds, "s", "N/A", Some("NA"), &off).unwrap();
        assert_eq!(text_cells(&lit.dataset, "s")[1].as_deref(), Some("NA"));
        let e = err_any(replace_with(&ds, "i", "-99", Some("NA"), &off));
        assert!(e.contains("'NA'") && e.contains("not an integer"), "{e}");
        // a missing value cannot be matched by replace (that is fill_na)
        let e = err_any(replace_with(&ds, "i", "NA", Some("0"), &off));
        assert!(e.contains("fill_na"), "{e}");
    }

    #[test]
    fn test_validate_replace_na_if() {
        use p2a_validation::{Check, Tol};
        // validation/r/cases/data_munging_t3.R, case data_munging_na_if: dplyr::na_if(x, sentinel)
        let (case, ds) = crate::test_support::load_case("data_munging_na_if");
        assert_eq!(dtype(&ds, "k"), DataType::Int64);
        let k = replace(&ds, "k", "3", "null").unwrap(); // the str_replace_value workflow
        let f = replace_with(&ds, "f", "1.25", None, &ReplaceOptions::default()).unwrap();
        let s = replace(&ds, "s", "c", "NA").unwrap();
        assert_eq!(dtype(&k, "k"), DataType::Int64);
        assert_eq!(
            (f.n_replaced, f.summary().as_str()),
            (2, "Replaced 2 values '1.25' with missing in column 'f'")
        );
        let present: Vec<f64> = text_cells(&s, "s")
            .iter()
            .map(|v| if v.is_some() { 1.0 } else { f64::NAN })
            .collect();
        Check::new(&case)
            .vector("k_na_if_3", &as_r(&num_cells(&k, "k")), Tol::exact())
            .vector(
                "f_na_if_1_25",
                &as_r(&num_cells(&f.dataset, "f")),
                Tol::exact(),
            )
            .vector("s_na_if_c", &present, Tol::exact())
            .finish();
        assert_eq!(as_r_names(&text_cells(&s, "s")), case.names("s_na_if_c"));
        // missing, not NaN: R's NA positions are polars nulls (2 original + 2 replaced)
        assert_eq!(k.df().column("k").unwrap().null_count(), 4);
        assert_eq!(f.dataset.df().column("f").unwrap().null_count(), 4);
        assert_eq!(s.df().column("s").unwrap().null_count(), 4);
    }

    #[test]
    fn test_replace_typed_every_dtype() {
        // DATA-38 repro: replace on every dtype but String/Int64/Float64 returned the column unchanged
        // while the tool reported success (Int32 [1, -99, 3] stayed [1, -99, 3]).
        let date = |d: &[Option<i32>]| {
            Int32Chunked::from_slice_options("d".into(), d)
                .into_date()
                .into_column()
        };
        let cat = Column::new("c".into(), ["a", "b", "a"])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let ds = Dataset::new(
            DataFrame::new(vec![
                Column::new("i32".into(), [1i32, -99, 3]),
                Column::new("f32".into(), [1.0f32, -99.0, 3.0]),
                Column::new("i16".into(), [1i16, -99, 3]),
                Column::new("u32".into(), [1u32, 99, 3]),
                Column::new("f32b".into(), [0.1f32, 0.25, 0.1]),
                Column::new("u8".into(), [1u8, 99, 3]),
                Column::new("b".into(), [Some(true), Some(false), None]),
                date(&[Some(18262), Some(18263), None]), // 2020-01-01, 2020-01-02 (R as.numeric(as.Date(..)))
                cat,
            ])
            .unwrap(),
        );
        assert_eq!(
            num_cells(&replace(&ds, "i32", "-99", "0").unwrap(), "i32"),
            [Some(1.0), Some(0.0), Some(3.0)]
        );
        let run = |c: &str, old: &str, new: Option<&str>| {
            replace_with(&ds, c, old, new, &ReplaceOptions::default()).unwrap()
        };
        for c in ["i32", "f32", "i16"] {
            let o = run(c, "-99", Some("0"));
            assert_eq!(
                (o.n_replaced, num_cells(&o.dataset, c)),
                (1, vec![Some(1.0), Some(0.0), Some(3.0)]),
                "{c}"
            );
            assert_eq!(dtype(&o.dataset, c), dtype(&ds, c), "{c}");
        }
        assert_eq!(
            run("i32", "-99", Some("0")).summary(),
            "Replaced 1 value '-99' with '0' in column 'i32'"
        );
        let u = run("u32", "99", None);
        assert_eq!(
            (dtype(&u.dataset, "u32"), num_cells(&u.dataset, "u32")),
            (DataType::UInt32, vec![Some(1.0), None, Some(3.0)])
        );
        // a Float32 cell 0.1 widens to 0.10000000149011612, not to the double 0.1
        let f = run("f32b", "0.1", Some("0.5"));
        assert_eq!(
            (f.n_replaced, dtype(&f.dataset, "f32b")),
            (2, DataType::Float32)
        );
        assert_eq!(
            f.dataset
                .df()
                .column("f32b")
                .unwrap()
                .f32()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [Some(0.5f32), Some(0.25), Some(0.5)]
        );
        let e = err_any(replace_with(
            &ds,
            "u8",
            "99",
            Some("300"),
            &ReplaceOptions::default(),
        ));
        assert!(
            e.contains("'300'") && e.contains("out of range") && e.contains("u8"),
            "{e}"
        );
        let b = run("b", "FALSE", Some("TRUE"));
        assert_eq!(
            b.dataset
                .df()
                .column("b")
                .unwrap()
                .bool()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [Some(true), Some(true), None]
        );
        let d = run("d", "2020-01-02", Some("2020-02-01")); // 2020-02-01 = 18293
        assert_eq!(dtype(&d.dataset, "d"), DataType::Date);
        assert_eq!(
            d.dataset
                .df()
                .column("d")
                .unwrap()
                .date()
                .unwrap()
                .physical()
                .into_iter()
                .collect::<Vec<_>>(),
            [Some(18262), Some(18293), None]
        );
        let c = run("c", "a", Some("z"));
        assert!(matches!(
            dtype(&c.dataset, "c"),
            DataType::Categorical(_, _)
        ));
        assert_eq!(
            (c.n_replaced, text_cells(&c.dataset, "c")),
            (
                2,
                vec![Some("z".into()), Some("b".into()), Some("z".into())]
            )
        );
        let e = err_any(replace_with(
            &ds,
            "i32",
            "x",
            Some("0"),
            &ReplaceOptions::default(),
        ));
        assert!(e.contains("'x'") && e.contains("i32"), "{e}");
        assert_eq!(
            run("i32", "7", Some("0")).summary(),
            "No value '7' found in column 'i32'; nothing was replaced"
        );
    }

    #[test]
    fn test_replace_unsupported_dtype_is_error() {
        // DATA-38: the old `_ => col.clone()` arm reported success on any other dtype
        let t = Int64Chunked::from_slice("t".into(), &[0, 1_000_000])
            .into_datetime(TimeUnit::Microseconds, None)
            .into_column();
        let ds = Dataset::new(DataFrame::new(vec![t]).unwrap());
        let e = err_text(replace(&ds, "t", "0", "1"));
        assert!(e.contains("'t'") && e.contains("datetime"), "{e}");
    }

    #[test]
    fn test_fill_constant_uint32_and_date() {
        // DATA-38 repro: Constant and Zero fills of UInt32, Int16 and Date columns left the nulls in place.
        let ds = Dataset::new(
            DataFrame::new(vec![
                Column::new("u".into(), [Some(5u32), None, Some(7)]),
                Column::new("i16".into(), [Some(1i16), None, Some(3)]),
                Int32Chunked::from_slice_options("d".into(), &[Some(18262), None, Some(18263)])
                    .into_date()
                    .into_column(),
            ])
            .unwrap(),
        );
        let fill = |c: &str, v: &str| fill_na(&ds, Some(&[c]), FillStrategy::Constant(v.into()));
        let u = fill("u", "7").unwrap();
        assert_eq!(
            (dtype(&u, "u"), num_cells(&u, "u")),
            (DataType::UInt32, vec![Some(5.0), Some(7.0), Some(7.0)])
        );
        let i = fill("i16", "300").unwrap();
        assert_eq!(
            (dtype(&i, "i16"), num_cells(&i, "i16")),
            (DataType::Int16, vec![Some(1.0), Some(300.0), Some(3.0)])
        );
        let d = fill("d", "2021-06-30").unwrap(); // R: as.numeric(as.Date("2021-06-30")) = 18808
        assert_eq!(dtype(&d, "d"), DataType::Date);
        assert_eq!(
            d.df()
                .column("d")
                .unwrap()
                .date()
                .unwrap()
                .physical()
                .into_iter()
                .collect::<Vec<_>>(),
            [Some(18262), Some(18808), Some(18263)]
        );
        for (c, v, needle) in [
            ("i16", "70000", "out of range"),
            ("u", "-1", "out of range"),
            ("d", "30/06/2021", "YYYY-MM-DD"),
        ] {
            let e = err_text(fill(c, v));
            assert!(e.contains(needle) && e.contains(&format!("'{c}'")), "{e}");
        }
        let z = fill_na(&ds, Some(&["u", "i16"]), FillStrategy::Zero).unwrap();
        assert_eq!(
            (
                num_cells(&z, "u")[1],
                num_cells(&z, "i16")[1],
                dtype(&z, "u")
            ),
            (Some(0.0), Some(0.0), DataType::UInt32)
        );
        let e = err_text(fill_na(&ds, Some(&["d"]), FillStrategy::Zero)); // R: tidyr::replace_na(<Date>, 0) errors
        assert!(e.contains("'d'") && e.contains("date"), "{e}");
    }

    #[test]
    fn test_fill_zero_and_constant_every_dtype() {
        let cat = Column::new("c".into(), [Some("a"), None])
            .cast(&DataType::from_categories(Categories::global()))
            .unwrap();
        let ds = Dataset::new(
            DataFrame::new(vec![
                Column::new("f32".into(), [Some(1.5f32), None]),
                Column::new("i8".into(), [Some(1i8), None]),
                Column::new("u64".into(), [Some(1u64), None]),
                Column::new("b".into(), [Some(true), None]),
                Column::new("s".into(), [Some("x"), None]),
                cat,
            ])
            .unwrap(),
        );
        let z = fill_na(&ds, None, FillStrategy::Zero).unwrap(); // categorical: skipped, no zero label
        assert_eq!(
            [
                num_cells(&z, "f32")[1],
                num_cells(&z, "i8")[1],
                num_cells(&z, "u64")[1],
                num_cells(&z, "b")[1]
            ],
            [Some(0.0); 4]
        );
        assert_eq!(
            (
                dtype(&z, "f32"),
                dtype(&z, "i8"),
                dtype(&z, "u64"),
                dtype(&z, "b")
            ),
            (
                DataType::Float32,
                DataType::Int8,
                DataType::UInt64,
                DataType::Boolean
            )
        );
        assert_eq!(text_cells(&z, "s")[1].as_deref(), Some(""));
        assert_eq!(z.df().column("c").unwrap().null_count(), 1);
        let e = err_text(fill_na(&ds, Some(&["c"]), FillStrategy::Zero)); // named: an error, not a skip
        assert!(e.contains("'c'") && e.contains("fill_na zero"), "{e}");
        let c = fill_na(&ds, Some(&["c"]), FillStrategy::Constant("z".into())).unwrap();
        assert!(matches!(dtype(&c, "c"), DataType::Categorical(_, _)));
        assert_eq!(
            text_cells(&c, "c"),
            [Some("a".to_string()), Some("z".to_string())]
        );
        let b = fill_na(&ds, Some(&["b"]), FillStrategy::Constant("F".into())).unwrap(); // R as.logical("F")
        assert_eq!(
            b.df().column("b").unwrap().bool().unwrap().get(1),
            Some(false)
        );
    }

    #[test]
    fn test_regex_helpers_match_stringr() {
        // DATA-42: the regex helpers had no value tests.
        // x <- c("ann@ex.org, bob@ex.com", NA, "no email", "tim@a.b.c")
        let ds = Dataset::new(
            df!("x" => [Some("ann@ex.org, bob@ex.com"), None, Some("no email"), Some("tim@a.b.c")])
                .unwrap(),
        );
        // stringr::str_match(x, "([a-z]+)@([a-z.]+)")[, 3]: "ex.org" NA NA "a.b.c"
        let m = regex_extract(&ds, "x", "([a-z]+)@([a-z.]+)", "dom", 2).unwrap();
        assert_eq!(
            split_col(&m, "dom"),
            [Some("ex.org".into()), None, None, Some("a.b.c".into())]
        );
        // str_extract_all(x, "[a-z]+@") pasted with ";": "ann@;bob@" NA "" "tim@"
        let a = regex_extract_all(&ds, "x", "[a-z]+@", "users", Some(";")).unwrap();
        assert_eq!(
            split_col(&a, "users"),
            [
                Some("ann@;bob@".into()),
                None,
                Some(String::new()),
                Some("tim@".into())
            ]
        );
        // str_count(x, "@"): 2 NA 0 1
        let c = regex_count(&ds, "x", "@", "n").unwrap();
        assert_eq!(
            c.df()
                .column("n")
                .unwrap()
                .i64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [Some(2), None, Some(0), Some(1)]
        );
        // str_replace_all(x, "([a-z]+)@", "<\\1>") (p2a writes the group as $1)
        let r = regex_replace(&ds, "x", "([a-z]+)@", "<$1>").unwrap();
        assert_eq!(
            split_col(&r, "x"),
            [
                Some("<ann>ex.org, <bob>ex.com".into()),
                None,
                Some("no email".into()),
                Some("<tim>a.b.c".into())
            ]
        );
    }

    #[test]
    fn test_fill_mode() {
        // DATA-28 (core): 'mode' was advertised but did not exist. R: names(which.max(table(x))), ties to
        // the first level in R's sorted order: table(c(TRUE, FALSE)) gives "FALSE" (R 4.3.3).
        let ds = Dataset::new(
            DataFrame::new(vec![
                Column::new("b".into(), [Some(true), None, Some(false), Some(true)]),
                Column::new("tie".into(), [Some(true), Some(false), None, None]),
                Column::new("u".into(), [Some(5u32), Some(7), None, Some(7)]),
                Int32Chunked::from_slice_options(
                    "d".into(),
                    &[Some(18263), None, Some(18262), Some(18263)],
                )
                .into_date()
                .into_column(),
                Column::new("c".into(), [Some("b"), Some("a"), None, Some("b")])
                    .cast(&DataType::from_categories(Categories::global()))
                    .unwrap(),
            ])
            .unwrap(),
        );
        let o = fill_na_with(&ds, None, &FillStrategy::Mode).unwrap();
        let df = o.dataset.df();
        let bools = |c: &str| {
            df.column(c)
                .unwrap()
                .bool()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bools("b"),
            [Some(true), Some(true), Some(false), Some(true)]
        );
        assert_eq!(
            bools("tie"),
            [Some(true), Some(false), Some(false), Some(false)]
        );
        assert_eq!(
            (dtype(&o.dataset, "u"), num_cells(&o.dataset, "u")[2]),
            (DataType::UInt32, Some(7.0))
        );
        assert_eq!(
            df.column("d").unwrap().date().unwrap().physical().get(1),
            Some(18263)
        );
        assert_eq!(text_cells(&o.dataset, "c")[2].as_deref(), Some("b"));
        let values: Vec<(&str, &str)> = o
            .fill_values
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        assert_eq!(
            values,
            [
                ("b", "TRUE"),
                ("tie", "FALSE"),
                ("u", "7"),
                ("d", "2020-01-02"),
                ("c", "b")
            ]
        );
        assert_eq!(
            o.filled.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
            [1, 2, 1, 1, 1]
        );
        let e = err_text(fill_na(
            &Dataset::new(df!("z" => [None::<i64>, None]).unwrap()),
            None,
            FillStrategy::Mode,
        ));
        assert!(
            e.contains("'z'") && e.contains("no observed values") && e.contains("mode"),
            "{e}"
        );
    }

    #[test]
    fn test_validate_fill_mode() {
        use p2a_validation::{Check, Tol};
        // validation/r/cases/data_munging_t3.R, case data_munging_fill_mode:
        // replace_na(x, names(which.max(table(x)))); k ties 3 (first in the data) and 1 -> 1; s ties a, b, c -> "a"
        let (case, ds) = crate::test_support::load_case("data_munging_fill_mode");
        let o = fill_na_with(&ds, Some(&["k", "f", "s"]), &FillStrategy::Mode).unwrap();
        assert_eq!(dtype(&o.dataset, "k"), DataType::Int64); // the mode keeps the integer type
        Check::new(&case)
            .vector("k_filled", &as_r(&num_cells(&o.dataset, "k")), Tol::exact())
            .vector("f_filled", &as_r(&num_cells(&o.dataset, "f")), Tol::exact())
            .scalar(
                "k_mode",
                o.fill_values[0].1.parse::<f64>().unwrap(),
                Tol::exact(),
            )
            .scalar(
                "f_mode",
                o.fill_values[1].1.parse::<f64>().unwrap(),
                Tol::exact(),
            )
            .finish();
        assert_eq!(
            as_r_names(&text_cells(&o.dataset, "s")),
            case.names("s_filled")
        );
        assert_eq!(case.names("s_mode"), [o.fill_values[2].1.clone()]);
        assert_eq!(
            o.summary(),
            "Filled 6 missing values in 3 columns: k (2, with 1), f (2, with 2.5), s (2, with 'a')"
        );
    }

    #[test]
    fn test_fill_strategy_kind_parses_aliases_and_rejects_unknown() {
        FillStrategyKind::choices().validate().unwrap();
        for (s, k) in [
            ("MEAN", FillStrategyKind::Mean),
            ("Forward-Fill", FillStrategyKind::Forward),
            ("ffill", FillStrategyKind::Forward),
            ("LOCF", FillStrategyKind::Forward),
            ("bfill", FillStrategyKind::Backward),
            ("most frequent", FillStrategyKind::Mode),
            ("Mode", FillStrategyKind::Mode),
            ("constant", FillStrategyKind::Constant),
        ] {
            assert_eq!(s.parse::<FillStrategyKind>().unwrap(), k, "{s}");
        }
        let e = "interpolate"
            .parse::<FillStrategyKind>()
            .unwrap_err()
            .to_string();
        assert!(e.contains("unknown fill strategy 'interpolate'; expected one of: mean, median, mode, forward, backward, zero, constant"), "{e}");
    }

    #[test]
    fn test_fill_strategy_parse() {
        // DATA-28 / LLM-12: keywords are case-insensitive, the constant keeps its case, unknown is an error
        assert!(matches!(
            FillStrategy::parse("Mode", None).unwrap(),
            FillStrategy::Mode
        ));
        assert!(
            matches!(FillStrategy::parse("constant", Some("Unknown")).unwrap(), FillStrategy::Constant(v) if v == "Unknown")
        );
        for (s, v, needle) in [
            ("interpolate", None, "unknown fill strategy 'interpolate'"),
            ("constant", None, "needs a value"),
            (
                "mean",
                Some("3"),
                "only used with strategy 'constant', not 'mean'",
            ),
        ] {
            let e = err_any(FillStrategy::parse(s, v));
            assert!(e.contains(needle), "{s}: {e}");
        }
        assert_eq!(FillStrategy::Mode.kind(), FillStrategyKind::Mode);
    }

    #[test]
    fn test_fill_na_with_skips_columns_that_cannot_take_the_constant() {
        // DATA-28: with no columns named, a text constant failed on the first numeric column
        let ds = Dataset::new(
            DataFrame::new(vec![
                Column::new("name".into(), [Some("Ann"), None]),
                Column::new("age".into(), [None, Some(30i64)]),
                Int32Chunked::from_slice_options("when".into(), &[Some(18262), None])
                    .into_date()
                    .into_column(),
            ])
            .unwrap(),
        );
        let o = fill_na_with(&ds, None, &FillStrategy::Constant("Unknown".into())).unwrap();
        assert_eq!(
            text_cells(&o.dataset, "name"),
            [Some("Ann".to_string()), Some("Unknown".to_string())]
        );
        assert_eq!(o.dataset.df().column("age").unwrap().null_count(), 1);
        assert_eq!(
            o.skipped
                .iter()
                .map(|(c, _)| c.as_str())
                .collect::<Vec<_>>(),
            ["age", "when"]
        );
        assert_eq!(
            o.summary(),
            "Filled 1 missing value in 1 column: name (1, with 'Unknown'); skipped 2 columns: \
            age (Invalid value 'Unknown' for column 'age': not an integer (the column is i64)), \
            when (Invalid value 'Unknown' for column 'when': not a date in YYYY-MM-DD form (the column is date))"
        );
        assert!(fill_na(&ds, None, FillStrategy::Constant("Unknown".into())).is_ok()); // was "Cannot parse as integer"
        assert!(
            err_text(fill_na(
                &ds,
                Some(&["age"]),
                FillStrategy::Constant("Unknown".into())
            ))
            .contains("'age'")
        );
    }
}
