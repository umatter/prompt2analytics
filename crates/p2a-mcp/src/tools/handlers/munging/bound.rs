//! Upper bounds of what the dataset quotas will charge for the dataset a
//! munging tool is about to store, and the admission of that dataset against
//! the quotas before the tool builds it.
//!
//! The governor checks the quotas after a call, on what the call stored. A
//! tool whose result can be far larger than its input (a separator or a
//! replacement written into every row, a list that repeats rows or columns,
//! a type that widens every value) would allocate all of it before that
//! check. Such a tool therefore predicts, from the row count, the input's
//! column widths and the request, an upper bound of what the quotas will
//! charge for the dataset it stores (`session::datasets_bytes_of`), and is
//! refused before it builds anything when the session's datasets, with that
//! dataset in place of any of the same name, would pass the session quota,
//! or the server's would pass the total (`session::admit_charge`). The bound
//! covers what is stored, not the working memory of the kernel that builds
//! it, except that a sample with replacement is charged at least the row
//! indices it draws. What a sample with replacement copies of a text column
//! depends on the rows it draws: it checks the rest of its result first
//! without reserving it (`session::charge_fits`), draws the rows, and is
//! admitted with their text counted. Without installed quotas (stdio, the
//! off profile, HTTP without quotas) nothing is predicted: [`active`] is
//! false.

use std::collections::HashSet;

use p2a_core::data::munging::{
    AggFn, AggSpec, FillStrategy, JoinShape, MutateExpr, parse_cast_dtype,
};
use p2a_core::polars::prelude::*;
use rmcp::ErrorData as McpError;

use crate::session::{self, VIEW_CHARGE_BYTES};

/// The longest String or Binary value polars' `View` holds inline: a
/// longer one keeps its bytes in a buffer as well.
const VIEW_INLINE_BYTES: u64 = 12;

/// Rows between two cooperative-cancellation checkpoints in the passes
/// over a column below.
const CHECKPOINT_EVERY: usize = 65_536;

/// Whether the dataset quotas of a governed call are installed on this
/// thread, so that a tool bounds its result before building it.
pub(super) fn active() -> bool {
    session::dataset_admission_active()
}

/// Admit the dataset `tool` is about to store as `result_name`, charged
/// at most `bytes`, or refuse it. The refusal says what makes the result
/// that large (`cause`, naming the request parameters), which quota it
/// would pass, and what to change (`advice`, the tool's own options; a
/// refusal by the session quota adds "start a new session", one by the
/// server total "try again later"); the governor answers it as
/// `quota_exceeded`.
pub(super) async fn admit(
    tool: &str,
    result_name: &str,
    bytes: u64,
    cause: impl FnOnce() -> String,
    advice: &[&str],
) -> Result<(), McpError> {
    let Err(shortfall) = session::admit_charge(result_name, bytes).await else {
        return Ok(());
    };
    Err(refusal(tool, bytes, "", &shortfall, cause(), advice))
}

/// [`admit`] for a part of the result, `bytes` at most for all of it but
/// `left_out`, without admitting it: a tool checks this before work that
/// grows with its result (drawing a sample's rows) and admits the whole
/// result once that work tells its size.
pub(super) async fn refuse_unless_part_fits(
    tool: &str,
    result_name: &str,
    bytes: u64,
    left_out: &str,
    cause: impl FnOnce() -> String,
    advice: &[&str],
) -> Result<(), McpError> {
    let Err(shortfall) = session::charge_fits(result_name, bytes).await else {
        return Ok(());
    };
    let part = format!(" without {left_out}");
    Err(refusal(tool, bytes, &part, &shortfall, cause(), advice))
}

/// The quota refusal of [`admit`] and [`refuse_unless_part_fits`].
fn refusal(
    tool: &str,
    bytes: u64,
    part: &str,
    shortfall: &session::QuotaShortfall,
    cause: String,
    advice: &[&str],
) -> McpError {
    tracing::info!(
        tool,
        requested = bytes,
        limit = shortfall.limit,
        per_session = shortfall.per_session,
        "munging result refused by a dataset quota before it was built"
    );
    // A new session has a quota of its own; the server's total frees up as
    // other sessions end.
    let last = if shortfall.per_session {
        "start a new session"
    } else {
        "try again later"
    };
    let options = if advice.is_empty() {
        last.to_string()
    } else {
        format!("{} or {last}", advice.join(", "))
    };
    session::quota_refusal(format!(
        "{tool} was not run: {cause} would make its result take up to {}{part}, so {}. \
         Nothing was built; {options}.",
        session::fmt_bytes(bytes),
        shortfall.outcome()
    ))
}

/// A cooperative-cancellation checkpoint every [`CHECKPOINT_EVERY`] rows
/// of a pass: the call's deadline or cancellation, as an error text.
pub(super) fn checkpoint(row: usize) -> Result<(), String> {
    if row.is_multiple_of(CHECKPOINT_EVERY) {
        p2a_core::resource::checkpoint().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// What the dataset a tool stores will be charged, at most, built up
/// column by column: the dataset's structure, each column's structure
/// (its name included) and each column's values.
#[derive(Debug)]
pub(super) struct Bound {
    /// The result's rows.
    rows: u64,
    /// Arrays a column of the result is split into, at most (a validity
    /// bitmap costs up to two bytes more per array).
    chunks: u64,
    bytes: u64,
}

impl Bound {
    /// A result of `rows` rows, built from frames whose columns are split
    /// into at most `chunks` arrays, with no column yet.
    pub(super) fn new(rows: u64, chunks: u64) -> Self {
        Self {
            rows,
            // A derived column can combine the arrays of two inputs.
            chunks: chunks.saturating_mul(2).saturating_add(1),
            bytes: session::structure_bytes(std::iter::empty()),
        }
    }

    /// A result of `df`'s rows that keeps every column of `df` but those
    /// named in `replaced` (polars puts a new column whose name `df` has
    /// in that column's place), each charged as the quotas charge it now.
    pub(super) fn keeping(df: &DataFrame, replaced: &[&str]) -> Self {
        let replaced: HashSet<&str> = replaced.iter().copied().collect();
        let mut bound = Self::new(df.height() as u64, df.max_n_chunks() as u64);
        for column in df.get_columns() {
            if !replaced.contains(column.name().as_str()) {
                bound.column(
                    column.name(),
                    session::series_bytes(column.as_materialized_series()),
                );
            }
        }
        bound
    }

    /// What the dataset is charged at most.
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The result's rows.
    pub(super) fn rows(&self) -> u64 {
        self.rows
    }

    /// Bytes that belong to no single column (text that the result's
    /// columns share out, or the values of a column added in parts).
    pub(super) fn add(&mut self, bytes: u64) {
        self.bytes = self.bytes.saturating_add(bytes);
    }

    /// A column named `name` whose values are charged at most `values`.
    pub(super) fn column(&mut self, name: &str, values: u64) {
        self.add(session::column_structure_bytes(name).saturating_add(values));
    }

    /// A column of the result's rows of fixed-width values of `width`
    /// bytes each, nulls allowed.
    pub(super) fn fixed(&mut self, name: &str, width: u64) {
        let values = width
            .saturating_mul(self.rows)
            .saturating_add(validity(self.rows, self.chunks));
        self.column(name, values);
    }

    /// A String column of the result's rows holding at most `text` bytes
    /// of text, nulls allowed.
    pub(super) fn string(&mut self, name: &str, text: u64) {
        let values = string_values(self.rows, text, self.chunks);
        self.column(name, values);
    }

    /// A column of the result's rows gathered from `source` (taken by row
    /// index) as `how` takes them, see [`gathered_values`].
    pub(super) fn gathered(&mut self, name: &str, source: &Column, how: Gather) {
        let values = gathered_values(source.as_materialized_series(), self.rows, self.chunks, how);
        self.column(name, values);
    }

    /// [`Bound::gathered`] of each source row at most `times` times: the
    /// smaller of that bound and the one for any rows (`Gather::Any`), as
    /// both hold.
    pub(super) fn gathered_at_most(&mut self, name: &str, source: &Column, times: u64) {
        let s = source.as_materialized_series();
        let repeated = gathered_values(s, self.rows, self.chunks, Gather::Repeat(times));
        let any = gathered_values(s, self.rows, self.chunks, Gather::Any);
        self.column(name, repeated.min(any));
    }
}

/// How a tool gathers a source column's rows into its result, which
/// bounds the bytes of the String or Binary values it gathers and the
/// inner values of the lists: polars copies every value it gathers from
/// a String column of several arrays, and shares a single array's
/// buffers with each of the arrays it splits a gather into.
#[derive(Debug, Clone, Copy)]
pub(super) enum Gather {
    /// Each source row at most once (a sample without replacement, a
    /// group's key or first value).
    Once,
    /// Each source row at most this many times (a melt's id columns, a
    /// join's columns of one side).
    Repeat(u64),
    /// Each source row once, and each of this many nulls filled with
    /// another row's value (a forward or backward fill of a nested column;
    /// a String or Binary column's fill is counted exactly, see
    /// [`filled_text`]).
    Fill(u64),
    /// Any rows, any number of times (a sample with replacement whose rows
    /// are not drawn yet).
    Any,
}

/// A validity bitmap of `rows` values in at most `chunks` arrays: a byte
/// per eight values, and up to two more per array (a slice that starts
/// inside a byte).
fn validity(rows: u64, chunks: u64) -> u64 {
    rows.div_ceil(8).saturating_add(chunks.saturating_mul(2))
}

/// `rows` String values holding at most `text` bytes of text: a view
/// each (as the quotas charge it), the text and a validity bitmap.
fn string_values(rows: u64, text: u64, chunks: u64) -> u64 {
    VIEW_CHARGE_BYTES
        .saturating_mul(rows)
        .saturating_add(text)
        .saturating_add(validity(rows, chunks))
}

/// The bytes a String value of `len` bytes keeps out of its view.
fn out_of_line(len: usize) -> u64 {
    let len = len as u64;
    if len > VIEW_INLINE_BYTES { len } else { 0 }
}

/// The width in bytes of one value of a fixed-width type (not Boolean).
fn fixed_width(dtype: &DataType) -> Option<u64> {
    Some(match dtype {
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Datetime(_, _)
        | DataType::Duration(_)
        | DataType::Time => 8,
        DataType::Int128 | DataType::UInt128 | DataType::Decimal(_, _) => 16,
        _ => return None,
    })
}

/// What `rows` values of a fixed-width type or Boolean are charged at
/// most, nulls allowed; `None` for other types.
fn fixed_values(dtype: &DataType, rows: u64, chunks: u64) -> Option<u64> {
    let validity = validity(rows, chunks);
    if *dtype == DataType::Boolean {
        // The values are a bitmap too.
        return Some(validity.saturating_mul(2));
    }
    fixed_width(dtype).map(|w| w.saturating_mul(rows).saturating_add(validity))
}

/// The bytes of each code of a Categorical or Enum type.
fn code_width(dtype: &DataType) -> u64 {
    match dtype.cat_physical() {
        Ok(CategoricalPhysical::U8) => 1,
        Ok(CategoricalPhysical::U16) => 2,
        _ => 4,
    }
}

/// The text of a String or Binary column: every value's bytes, inline
/// values and null slots included. The quotas charge an array's text as
/// the smaller of its buffers and its values' bytes, so an array built
/// from these values, or sharing their buffers, is charged no more.
/// Zero for other types.
pub(super) fn text_bytes(s: &Series) -> u64 {
    let mut text = 0u64;
    match s.dtype() {
        DataType::String => {
            if let Ok(ca) = s.str() {
                for arr in ca.downcast_iter() {
                    text = text.saturating_add(arr.total_bytes_len() as u64);
                }
            }
        }
        DataType::Binary => {
            if let Ok(ca) = s.binary() {
                for arr in ca.downcast_iter() {
                    text = text.saturating_add(arr.total_bytes_len() as u64);
                }
            }
        }
        _ => {}
    }
    text
}

/// The bytes of the longest value of a String or Binary column (0 for
/// other types).
fn longest_value(s: &Series) -> u64 {
    let longest = match s.dtype() {
        DataType::String => s.str().ok().and_then(|ca| {
            ca.downcast_iter()
                .flat_map(|a| a.views().iter().map(|v| v.length))
                .max()
        }),
        DataType::Binary => s.binary().ok().and_then(|ca| {
            ca.downcast_iter()
                .flat_map(|a| a.views().iter().map(|v| v.length))
                .max()
        }),
        _ => None,
    };
    u64::from(longest.unwrap_or(0))
}

/// The text of `rows` values gathered from the String or Binary column
/// `s` as `how` takes them, at most: the bytes of the values gathered.
/// That bounds the charge whether the gather copies them or shares `s`'s
/// buffers, also when polars splits it into several arrays that each
/// share all of them.
fn gathered_text(s: &Series, rows: u64, how: Gather) -> u64 {
    let text = text_bytes(s);
    match how {
        Gather::Once => text,
        Gather::Repeat(times) => text.saturating_mul(times),
        Gather::Fill(nulls) => text.saturating_add(nulls.saturating_mul(longest_value(s))),
        Gather::Any => rows.saturating_mul(longest_value(s)),
    }
}

/// The inner values `rows` lists gathered from a list column take, at
/// most, given its `inner` value count and `longest` list.
fn gathered_inner(inner: u64, longest: u64, rows: u64, how: Gather) -> (u64, Gather) {
    match how {
        Gather::Once => (inner, Gather::Once),
        Gather::Repeat(times) => (inner.saturating_mul(times), Gather::Repeat(times)),
        Gather::Fill(nulls) => (
            inner.saturating_add(nulls.saturating_mul(longest)),
            Gather::Any,
        ),
        Gather::Any => (rows.saturating_mul(longest), Gather::Any),
    }
}

/// What `rows` values gathered from `s` as `how` takes them are charged
/// at most: a fixed-width value its width; a String or Binary value its
/// view and the text of [`gathered_text`] (polars' compaction of a
/// gathered array only shrinks its buffers); a Categorical or Enum value
/// its code, with `s`'s labels; nested values recursively; and validity.
/// A type a dataset rarely holds is charged its whole column for every
/// row.
fn gathered_values(s: &Series, rows: u64, chunks: u64, how: Gather) -> u64 {
    let dtype = s.dtype();
    if let Some(values) = fixed_values(dtype, rows, chunks) {
        return values;
    }
    let whole_column_per_row = || {
        session::series_bytes(s)
            .saturating_add(16)
            .saturating_mul(rows.max(1))
    };
    let validity = validity(rows, chunks);
    match dtype {
        DataType::String | DataType::Binary => {
            string_values(rows, gathered_text(s, rows, how), chunks)
        }
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let labels = session::series_bytes(s).saturating_sub(s.estimated_size() as u64);
            code_width(dtype)
                .saturating_mul(rows)
                .saturating_add(labels)
                .saturating_add(validity)
        }
        DataType::List(_) => {
            let Ok(ca) = s.list() else {
                return whole_column_per_row();
            };
            let longest = ca
                .downcast_iter()
                .flat_map(|a| a.offsets().lengths())
                .max()
                .unwrap_or(0) as u64;
            let inner = ca.get_inner();
            let (values, inner_how) = gathered_inner(inner.len() as u64, longest, rows, how);
            rows.saturating_add(chunks)
                .saturating_mul(8)
                .saturating_add(validity)
                .saturating_add(gathered_values(&inner, values, chunks, inner_how))
        }
        DataType::Array(_, width) => match s.array() {
            Ok(ca) => validity.saturating_add(gathered_values(
                &ca.get_inner(),
                rows.saturating_mul(*width as u64),
                chunks,
                match how {
                    Gather::Fill(nulls) => Gather::Fill(nulls.saturating_mul(*width as u64)),
                    other => other,
                },
            )),
            Err(_) => whole_column_per_row(),
        },
        DataType::Struct(_) => match s.struct_() {
            Ok(ca) => ca.fields_as_series().iter().fold(validity, |acc, field| {
                acc.saturating_add(gathered_values(field, rows, chunks, how))
            }),
            Err(_) => whole_column_per_row(),
        },
        _ => whole_column_per_row(),
    }
}

/// What `rows` null values of type `dtype` are charged at most: the
/// column polars fills with nulls where a row-bound dataset lacks it.
fn null_values(dtype: &DataType, rows: u64, chunks: u64) -> u64 {
    if let Some(values) = fixed_values(dtype, rows, chunks) {
        return values;
    }
    let values = match dtype {
        DataType::String | DataType::Binary => VIEW_CHARGE_BYTES.saturating_mul(rows),
        DataType::Categorical(_, _) => code_width(dtype).saturating_mul(rows),
        DataType::Enum(frozen, _) => code_width(dtype)
            .saturating_mul(rows)
            .saturating_add(session::frozen_label_bytes(frozen)),
        DataType::List(_) => rows.saturating_add(chunks).saturating_mul(8),
        DataType::Array(inner, width) => {
            null_values(inner, rows.saturating_mul(*width as u64), chunks)
        }
        DataType::Struct(fields) => fields.iter().fold(0u64, |acc, field| {
            acc.saturating_add(null_values(field.dtype(), rows, chunks))
        }),
        _ => 16u64.saturating_mul(rows),
    };
    values.saturating_add(validity(rows, chunks))
}

// Each tool's bound below is `None` when the request names a column the
// tool refuses (missing, or of a type it does not take): the tool then
// reports its own error and builds nothing.

/// `str_concat`: `columns` (repeats allowed) joined by `separator`,
/// nulls skipped, into `new_column`. A row holds at most every listed
/// value and one separator between two of them.
pub(super) fn str_concat(
    df: &DataFrame,
    columns: &[&str],
    new_column: &str,
    separator: &str,
) -> Option<u64> {
    let mut text = 0u64;
    for name in columns {
        let column = df.column(name).ok()?;
        if column.dtype() != &DataType::String {
            return None;
        }
        text = text.saturating_add(text_bytes(column.as_materialized_series()));
    }
    let mut bound = Bound::keeping(df, &[new_column]);
    let separators = (columns.len() as u64)
        .saturating_sub(1)
        .saturating_mul(separator.len() as u64)
        .saturating_mul(bound.rows());
    bound.string(new_column, text.saturating_add(separators));
    Some(bound.bytes())
}

/// `str_replace_value` on a String column: each value equal to `old`
/// becomes `new`, every value is copied. Returns the bound and the
/// number of values replaced. (A numeric column keeps its width, and a
/// column of another type is kept: `None`.)
pub(super) fn replace_value(
    df: &DataFrame,
    column: &str,
    old: &str,
    new: &str,
) -> Result<Option<(u64, u64)>, String> {
    let Some(source) = df.column(column).ok() else {
        return Ok(None);
    };
    let Ok(ca) = source.str() else {
        return Ok(None);
    };
    let mut matches = 0u64;
    for (row, value) in ca.into_iter().enumerate() {
        checkpoint(row)?;
        matches += u64::from(value == Some(old));
    }
    let text = text_bytes(source.as_materialized_series())
        .saturating_sub(matches.saturating_mul(old.len() as u64))
        .saturating_add(matches.saturating_mul(new.len() as u64));
    let mut bound = Bound::keeping(df, &[column]);
    bound.string(column, text);
    Ok(Some((bound.bytes(), matches)))
}

/// `str_regex_replace`: the replaced column holds at most `text` bytes
/// (`p2a_core::data::munging::regex_replace_bytes`).
pub(super) fn regex_replace(df: &DataFrame, column: &str, text: u64) -> u64 {
    let mut bound = Bound::keeping(df, &[column]);
    bound.string(column, text);
    bound.bytes()
}

/// `str_to_uppercase` (`upper`) or `str_to_lowercase`: the column with
/// its text after the case mapping, counted exactly (a mapping can take
/// up to three times a character's bytes).
pub(super) fn case_mapped(
    df: &DataFrame,
    column: &str,
    upper: bool,
) -> Result<Option<u64>, String> {
    let Some(ca) = df.column(column).ok().and_then(|c| c.str().ok()) else {
        return Ok(None);
    };
    let mut text = 0u64;
    for (row, value) in ca.into_iter().enumerate() {
        checkpoint(row)?;
        if let Some(s) = value {
            text = text.saturating_add(mapped_len(s, upper));
        }
    }
    let mut bound = Bound::keeping(df, &[column]);
    bound.string(column, text);
    Ok(Some(bound.bytes()))
}

/// The bytes of `s` mapped to upper (or lower) case character by
/// character, as Rust's and polars' case mappings map it.
fn mapped_len(s: &str, upper: bool) -> u64 {
    if s.is_ascii() {
        return s.len() as u64;
    }
    let bytes: usize = if upper {
        s.chars()
            .map(|c| c.to_uppercase().map(char::len_utf8).sum::<usize>())
            .sum()
    } else {
        s.chars()
            .map(|c| c.to_lowercase().map(char::len_utf8).sum::<usize>())
            .sum()
    };
    bytes as u64
}

/// `str_split` of `column` into `parts` columns `{prefix}_{i}`: each a
/// String column of every row, together holding at most the column's
/// text.
pub(super) fn split(df: &DataFrame, column: &str, prefix: &str, parts: usize) -> Option<u64> {
    let source = df.column(column).ok()?;
    if source.dtype() != &DataType::String {
        return None;
    }
    let names: Vec<String> = (0..parts).map(|i| format!("{prefix}_{i}")).collect();
    let replaced: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut bound = Bound::keeping(df, &replaced);
    for name in &names {
        bound.string(name, 0);
    }
    bound.add(text_bytes(source.as_materialized_series()));
    Some(bound.bytes())
}

/// `munge_fill_na` of `columns` (every column when `None`): a String
/// column filled with a constant is copied with each null replaced, a
/// numeric column filled with its mean or median becomes Float64, a
/// forward or backward fill copies the value each null takes (counted
/// exactly for a String or Binary column, see [`filled_text`]), and any
/// other fill keeps the column's type and width.
pub(super) fn fill_na(
    df: &DataFrame,
    columns: Option<&[&str]>,
    strategy: &FillStrategy,
) -> Result<Option<u64>, String> {
    let names: Vec<&str> = match columns {
        Some(names) => names.to_vec(),
        None => df
            .get_column_names()
            .into_iter()
            .map(|n| n.as_str())
            .collect(),
    };
    let mut bound = Bound::keeping(df, &names);
    for name in names {
        let Ok(column) = df.column(name) else {
            return Ok(None);
        };
        let dtype = column.dtype();
        match strategy {
            FillStrategy::Constant(_) | FillStrategy::Zero if dtype == &DataType::String => {
                let fill = match strategy {
                    FillStrategy::Constant(value) => value.len() as u64,
                    _ => 0,
                };
                let nulls = column.null_count() as u64;
                let text = text_bytes(column.as_materialized_series())
                    .saturating_add(nulls.saturating_mul(fill));
                bound.string(name, text);
            }
            FillStrategy::Mean | FillStrategy::Median if dtype.is_primitive_numeric() => {
                bound.fixed(name, 8);
            }
            FillStrategy::Forward | FillStrategy::Backward
                if matches!(dtype, DataType::String | DataType::Binary) =>
            {
                let forward = matches!(strategy, FillStrategy::Forward);
                let text = filled_text(column.as_materialized_series(), forward)?;
                bound.string(name, text);
            }
            FillStrategy::Forward | FillStrategy::Backward => {
                let nulls = column.null_count() as u64;
                bound.gathered(name, column, Gather::Fill(nulls))
            }
            _ => bound.column(name, session::series_bytes(column.as_materialized_series())),
        }
    }
    Ok(Some(bound.bytes()))
}

/// The text of the String or Binary column `s` after a forward (`forward`)
/// or backward fill, counted in one pass without building it: each value's
/// own bytes (a null slot's too), and for each null the bytes of the last
/// value before it (forward) or the first after it (backward). A null with
/// no such value stays null and adds nothing.
pub(super) fn filled_text(s: &Series, forward: bool) -> Result<u64, String> {
    let Some(ca) = as_views(s)? else {
        return Ok(text_bytes(s));
    };
    let mut text = 0u64;
    // The length of the value a null takes, once a value was passed.
    let mut fill: Option<u64> = None;
    let mut row = 0usize;
    let mut visit = |length: u32, valid: bool| -> Result<(), String> {
        checkpoint(row)?;
        row += 1;
        let length = u64::from(length);
        text = text.saturating_add(length);
        if valid {
            fill = Some(length);
        } else if let Some(fill) = fill {
            text = text.saturating_add(fill);
        }
        Ok(())
    };
    if forward {
        for arr in ca.downcast_iter() {
            for (i, view) in arr.views().iter().enumerate() {
                visit(view.length, arr.get(i).is_some())?;
            }
        }
    } else {
        for arr in ca.downcast_iter().rev() {
            for (i, view) in arr.views().iter().enumerate().rev() {
                visit(view.length, arr.get(i).is_some())?;
            }
        }
    }
    Ok(text)
}

/// The String or Binary column `s` as Binary, sharing its views and
/// buffers (`None` for another type).
fn as_views(s: &Series) -> Result<Option<BinaryChunked>, String> {
    Ok(match s.dtype() {
        DataType::String => Some(s.str().map_err(|e| e.to_string())?.as_binary()),
        DataType::Binary => Some(s.binary().map_err(|e| e.to_string())?.clone()),
        _ => None,
    })
}

/// `munge_mutate`: the column `new_column`, a constant on every row
/// (Float64 when it parses as a number, else String) or a Float64
/// computed from columns.
pub(super) fn mutate(df: &DataFrame, new_column: &str, expr: &MutateExpr) -> Option<u64> {
    // A column the expression names must exist (the tool reports it).
    match expr {
        MutateExpr::Arithmetic(a, _, b) => {
            df.column(a).ok()?;
            df.column(b).ok()?;
        }
        MutateExpr::ArithmeticConst(c, _, _) | MutateExpr::Function(_, c) => {
            df.column(c).ok()?;
        }
        MutateExpr::Constant(_) | MutateExpr::Copy(_) => {}
    }
    let mut bound = Bound::keeping(df, &[new_column]);
    match expr {
        MutateExpr::Constant(value) if value.parse::<f64>().is_err() => {
            let text = out_of_line(value.len()).saturating_mul(bound.rows());
            bound.string(new_column, text);
        }
        MutateExpr::Copy(source) => {
            let s = df.column(source).ok()?.as_materialized_series();
            bound.column(new_column, session::series_bytes(s));
        }
        MutateExpr::Constant(_)
        | MutateExpr::Arithmetic(..)
        | MutateExpr::ArithmeticConst(..)
        | MutateExpr::Function(..) => bound.fixed(new_column, 8),
    }
    Some(bound.bytes())
}

/// `munge_bin`: the String column `{column}_binned`, one label of at
/// most `longest_label` bytes per row.
pub(super) fn bin(df: &DataFrame, column: &str, longest_label: usize) -> Option<u64> {
    df.column(column).ok()?;
    let name = format!("{column}_binned");
    let mut bound = Bound::keeping(df, &[&name]);
    let text = out_of_line(longest_label).saturating_mul(bound.rows());
    bound.string(&name, text);
    Some(bound.bytes())
}

/// `munge_one_hot_encode`: one Int32 column `{column}_{value}` per
/// distinct value (each of them: which one `drop_first` drops depends
/// on the order polars lists them in). Returns the bound and the number
/// of distinct values.
pub(super) fn one_hot(df: &DataFrame, column: &str) -> Result<Option<(u64, usize)>, String> {
    let Ok(source) = df.column(column) else {
        return Ok(None);
    };
    // polars refuses the column here as the tool will: it reports that.
    let Ok(unique) = source.unique() else {
        return Ok(None);
    };
    let mut names = Vec::with_capacity(unique.len());
    for i in 0..unique.len() {
        checkpoint(i)?;
        let value = unique
            .get(i)
            .map(|v| format!("{v}"))
            .unwrap_or_else(|_| "null".to_string());
        names.push(format!("{column}_{value}"));
    }
    let replaced: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut bound = Bound::keeping(df, &replaced);
    for name in &names {
        bound.fixed(name, 4);
    }
    Ok(Some((bound.bytes(), names.len())))
}

/// `munge_concat`: the frames row-bound, each column of the union of
/// their columns as each frame holds it, or filled with nulls in a frame
/// that lacks it.
pub(super) fn concat(frames: &[&DataFrame]) -> u64 {
    let mut schema: Vec<(&str, &DataType)> = Vec::new();
    let mut seen = HashSet::new();
    for df in frames {
        for column in df.get_columns() {
            if seen.insert(column.name().as_str()) {
                schema.push((column.name().as_str(), column.dtype()));
            }
        }
    }
    let rows = frames
        .iter()
        .fold(0u64, |acc, df| acc.saturating_add(df.height() as u64));
    let chunks = frames
        .iter()
        .fold(0u64, |acc, df| acc.saturating_add(df.max_n_chunks() as u64));
    let mut bound = Bound::new(rows, chunks);
    for (name, _) in &schema {
        bound.column(name, 0);
    }
    for df in frames {
        for (name, dtype) in &schema {
            match df.column(name) {
                Ok(column) => bound.add(session::series_bytes(column.as_materialized_series())),
                Err(_) => bound.add(null_values(dtype, df.height() as u64, 1)),
            }
        }
    }
    bound.bytes()
}

/// `munge_melt`: every row once per value variable, the id columns
/// gathered, the variable names as String and the values as Float64.
pub(super) fn melt(
    df: &DataFrame,
    id_vars: &[&str],
    value_vars: &[&str],
    variable_name: &str,
    value_name: &str,
) -> Option<u64> {
    for v in value_vars {
        df.column(v).ok()?;
    }
    let height = df.height() as u64;
    let times = value_vars.len() as u64;
    let rows = height.saturating_mul(times);
    let mut bound = Bound::new(rows, df.max_n_chunks() as u64);
    for id in id_vars {
        bound.gathered(id, df.column(id).ok()?, Gather::Repeat(times));
    }
    let names = value_vars
        .iter()
        .fold(0u64, |acc, v| acc.saturating_add(out_of_line(v.len())));
    bound.string(variable_name, names.saturating_mul(height));
    bound.fixed(value_name, 8);
    Some(bound.bytes())
}

/// The bytes of one row index of polars (`IdxSize`): a sample with
/// replacement draws one per sampled row before it gathers them.
pub(super) const ROW_INDEX_BYTES: u64 = std::mem::size_of::<IdxSize>() as u64;

/// `munge_sample` of `rows` rows, with `replacement` or without: every
/// column gathered.
pub(super) fn sample(df: &DataFrame, rows: u64, replacement: bool) -> u64 {
    let how = if replacement {
        Gather::Any
    } else {
        Gather::Once
    };
    let mut bound = Bound::new(rows, df.max_n_chunks() as u64);
    for column in df.get_columns() {
        bound.gathered(column.name(), column, how);
    }
    bound.bytes()
}

/// Whether `df` has a String or Binary column, whose text a sample with
/// replacement can only bound once its rows are drawn.
pub(super) fn has_text(df: &DataFrame) -> bool {
    df.get_columns()
        .iter()
        .any(|c| matches!(c.dtype(), DataType::String | DataType::Binary))
}

/// [`sample`]'s bound of a sample with replacement of `rows` rows, with the
/// text of its String and Binary columns left out (their views counted):
/// every other column is bounded as [`sample_drawn`] bounds it, so this is
/// never above [`sample_drawn`], whichever rows are drawn.
pub(super) fn sample_without_text(df: &DataFrame, rows: u64) -> u64 {
    let mut bound = Bound::new(rows, df.max_n_chunks() as u64);
    for column in df.get_columns() {
        if matches!(column.dtype(), DataType::String | DataType::Binary) {
            bound.string(column.name(), 0);
        } else {
            bound.gathered(column.name(), column, Gather::Any);
        }
    }
    bound.bytes()
}

/// `munge_sample` of the drawn rows `drawn` (`p2a_core::data::munging::
/// sample_rows`): every column gathered as [`sample`] gathers it with
/// replacement, but a String or Binary column's text counted exactly, as
/// the bytes of the drawn rows' values.
pub(super) fn sample_drawn(df: &DataFrame, drawn: &IdxCa) -> Result<u64, String> {
    let mut bound = Bound::new(drawn.len() as u64, df.max_n_chunks() as u64);
    for column in df.get_columns() {
        if matches!(column.dtype(), DataType::String | DataType::Binary) {
            let text = drawn_text(column.as_materialized_series(), drawn)?;
            bound.string(column.name(), text);
        } else {
            bound.gathered(column.name(), column, Gather::Any);
        }
    }
    Ok(bound.bytes())
}

/// The bytes of the values of the String or Binary column `s` at the rows
/// `drawn` (each as often as it is drawn; a null slot's bytes too).
pub(super) fn drawn_text(s: &Series, drawn: &IdxCa) -> Result<u64, String> {
    let Some(ca) = as_views(s)? else {
        return Ok(0);
    };
    let arrays: Vec<_> = ca.downcast_iter().collect();
    // The first row of each array, to find the array a row is in.
    let mut starts = Vec::with_capacity(arrays.len());
    let mut first = 0usize;
    for arr in &arrays {
        starts.push(first);
        first += arr.len();
    }
    let mut text = 0u64;
    for (i, row) in drawn.iter().enumerate() {
        checkpoint(i)?;
        let Some(row) = row else { continue };
        let row = row as usize;
        let array = starts
            .partition_point(|&start| start <= row)
            .saturating_sub(1);
        let length = arrays
            .get(array)
            .and_then(|arr| arr.views().get(row - starts[array]))
            .map_or(0, |view| u64::from(view.length));
        text = text.saturating_add(length);
    }
    Ok(text)
}

/// `munge_join` of the predicted `shape`: every column of both frames
/// gathered (the right key columns too, which polars may drop), each row of
/// a side at most as often as the shape says it repeats, and a right
/// column whose name the left has renamed with `suffix`.
pub(super) fn join(left: &DataFrame, right: &DataFrame, shape: &JoinShape, suffix: &str) -> u64 {
    let chunks = (left.max_n_chunks() as u64).saturating_add(right.max_n_chunks() as u64);
    let mut bound = Bound::new(shape.rows, chunks);
    for column in left.get_columns() {
        bound.gathered_at_most(column.name(), column, shape.max_left_repeat);
    }
    for column in right.get_columns() {
        let times = shape.max_right_repeat;
        if left.column(column.name()).is_ok() {
            bound.gathered_at_most(&format!("{}{suffix}", column.name()), column, times);
        } else {
            bound.gathered_at_most(column.name(), column, times);
        }
    }
    bound.bytes()
}

/// `munge_group_by`: one row per group (at most the rows, and at most
/// the product of the `by` columns' distinct counts), the `by` columns
/// gathered and one column per aggregation. Returns the bound and the
/// group count it assumed.
pub(super) fn group_by(df: &DataFrame, by: &[&str], aggs: &[AggSpec]) -> Option<(u64, u64)> {
    let mut groups = 1u64;
    for name in by {
        let column = df.column(name).ok()?;
        // A key polars cannot count may still group (or the tool reports
        // its own error): bound it by the rows.
        let distinct = column.n_unique().unwrap_or(df.height()) as u64;
        groups = groups.saturating_mul(distinct);
    }
    let groups = groups.min(df.height() as u64);
    let mut bound = Bound::new(groups, df.max_n_chunks() as u64);
    for name in by {
        if let Ok(column) = df.column(name) {
            bound.gathered(name, column, Gather::Once);
        }
    }
    for spec in aggs {
        let column = df.column(&spec.column).ok()?;
        let name = spec.result_name();
        let dtype = column.dtype();
        let scalar = dtype == &DataType::Boolean
            || dtype == &DataType::String
            || fixed_width(dtype).is_some();
        match spec.agg_fn {
            AggFn::Count | AggFn::NUnique => bound.fixed(&name, 8),
            // A null String for a String column; otherwise a number of the
            // column's width, at least 8 bytes (a mean is Float64, a sum of
            // small integers Int64).
            AggFn::Sum | AggFn::Mean | AggFn::Median | AggFn::Std | AggFn::Var if scalar => {
                let width = if dtype == &DataType::String {
                    VIEW_CHARGE_BYTES
                } else {
                    fixed_width(dtype).unwrap_or(8).max(8)
                };
                bound.fixed(&name, width)
            }
            // One value of the group (each source value at most once),
            // or a type a dataset rarely holds.
            _ => bound.gathered(&name, column, Gather::Once),
        }
    }
    Some((bound.bytes(), groups))
}

/// A type `munge_cast` takes, by the core's own names for it, or `None`.
fn cast_target(name: &str) -> Option<DataType> {
    parse_cast_dtype(name).ok()
}

/// The longest text polars writes for one value of `dtype` cast to
/// String: integers in decimal, floats in their shortest form (ryu, at
/// most 24 bytes), temporal values as ISO text, generously for the rest.
fn cast_text_width(dtype: &DataType) -> usize {
    match dtype {
        DataType::Boolean => 5,
        DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16 => 6,
        DataType::Int32 | DataType::UInt32 => 11,
        DataType::Int64 | DataType::UInt64 => 20,
        DataType::Int128 | DataType::UInt128 => 40,
        DataType::Float32 => 16,
        DataType::Float64 => 24,
        DataType::Date => 16,
        DataType::Time => 32,
        _ => 64,
    }
}

/// `munge_cast`: each `(column, type)` pair cast in turn; a column cast
/// more than once is charged for its largest version, and a cast to the
/// type a column already has keeps it.
pub(super) fn cast(df: &DataFrame, casts: &[(&str, &str)]) -> Result<Option<u64>, String> {
    /// A cast column: its type now, the bound of its current version
    /// and of its largest one.
    struct Cast<'a> {
        name: &'a str,
        dtype: DataType,
        now: u64,
        largest: u64,
    }
    let rows = df.height() as u64;
    let chunks = df.max_n_chunks() as u64;
    let mut cast: Vec<Cast> = Vec::new();
    for (name, target) in casts {
        let (Ok(source), Some(target)) = (df.column(name), cast_target(target)) else {
            return Ok(None);
        };
        let index = match cast.iter().position(|c| c.name == *name) {
            Some(i) => i,
            None => {
                let charged = session::series_bytes(source.as_materialized_series());
                cast.push(Cast {
                    name,
                    dtype: source.dtype().clone(),
                    now: charged,
                    largest: charged,
                });
                cast.len() - 1
            }
        };
        let current = &cast[index];
        let next = if current.dtype == target {
            current.now
        } else if let Some(values) = fixed_values(&target, rows, chunks) {
            values
        } else {
            // To String: the column's own text while it has its own
            // type, else the widest text of the type it was cast to.
            let text = if &current.dtype == source.dtype() {
                cast_source_text(source.as_materialized_series())?
            } else {
                out_of_line(cast_text_width(&current.dtype)).saturating_mul(rows)
            };
            string_values(rows, text, chunks)
        };
        let entry = &mut cast[index];
        entry.dtype = target;
        entry.now = next;
        entry.largest = entry.largest.max(next);
    }
    let replaced: Vec<&str> = cast.iter().map(|c| c.name).collect();
    let mut bound = Bound::keeping(df, &replaced);
    for c in &cast {
        bound.column(c.name, c.largest);
    }
    Ok(Some(bound.bytes()))
}

/// The text a cast of the column `s` to String writes, at most: a String
/// or Binary column's own text, a Categorical or Enum column's label on
/// every row, and for other types the widest text of a value (when a
/// view cannot hold it) on every row.
fn cast_source_text(s: &Series) -> Result<u64, String> {
    match s.dtype() {
        DataType::String | DataType::Binary => Ok(text_bytes(s)),
        DataType::Categorical(_, mapping) => {
            let mapping = mapping.clone();
            codes_text(s, |code| mapping.cat_to_str(code).map_or(0, str::len))
        }
        DataType::Enum(frozen, _) => {
            let labels = frozen.categories().clone();
            codes_text(s, |code| {
                let code = code as usize;
                if code < labels.len() {
                    labels.value(code).len()
                } else {
                    0
                }
            })
        }
        dtype => {
            let rows = s.len() as u64;
            Ok(out_of_line(cast_text_width(dtype)).saturating_mul(rows))
        }
    }
}

/// The bytes of `label(code)` summed over the non-null codes of the
/// Categorical or Enum column `s`.
fn codes_text(s: &Series, label: impl Fn(u32) -> usize) -> Result<u64, String> {
    let physical = s.to_physical_repr();
    let mut text = 0u64;
    let mut add = |row: usize, code: Option<u32>| -> Result<(), String> {
        checkpoint(row)?;
        if let Some(code) = code {
            text = text.saturating_add(label(code) as u64);
        }
        Ok(())
    };
    let err = |e: PolarsError| e.to_string();
    match physical.dtype() {
        DataType::UInt8 => {
            for (row, code) in physical.u8().map_err(err)?.into_iter().enumerate() {
                add(row, code.map(u32::from))?;
            }
        }
        DataType::UInt16 => {
            for (row, code) in physical.u16().map_err(err)?.into_iter().enumerate() {
                add(row, code.map(u32::from))?;
            }
        }
        DataType::UInt32 => {
            for (row, code) in physical.u32().map_err(err)?.into_iter().enumerate() {
                add(row, code)?;
            }
        }
        _ => {
            let rows = s.len() as u64;
            return Ok(session::series_bytes(s).saturating_mul(rows.max(1)));
        }
    }
    Ok(text)
}

/// `munge_standardize` (`suffix` `standardized`) or normalize
/// (`normalized`): one Float64 column `{c}_{suffix}` per listed column.
pub(super) fn standardize(df: &DataFrame, columns: &[&str], suffix: &str) -> Option<u64> {
    for c in columns {
        df.column(c).ok()?;
    }
    let names: Vec<String> = columns.iter().map(|c| format!("{c}_{suffix}")).collect();
    let replaced: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut bound = Bound::keeping(df, &replaced);
    for name in &names {
        bound.fixed(name, 8);
    }
    Some(bound.bytes())
}

/// `munge_diff`: the column `new_column`, of `column`'s width or a
/// Float64 (a percent change, a difference of dates), whichever is
/// wider.
pub(super) fn diff(df: &DataFrame, column: &str, new_column: &str) -> Option<u64> {
    let width = fixed_width(df.column(column).ok()?.dtype())
        .unwrap_or(8)
        .max(8);
    let mut bound = Bound::keeping(df, &[new_column]);
    bound.fixed(new_column, width);
    Some(bound.bytes())
}
