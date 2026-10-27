//! Distinct values and combinations of dataset columns, counted for the pivot budgets
//! ([`super::Limits::check_pivot`]) up to the first count no budget can pass.
//!
//! polars' own counts take memory in proportion to the rows of the frame: `n_unique` of an
//! unsorted column sorts a copy of it, and grouping by several columns builds the row indices of
//! every group. [`count`] instead keeps one hash map entry per distinct value seen and stops once
//! it holds `stop` of them. One column is counted in one set, and a count that cannot pass stops
//! after reading only as many rows as it takes to be sure. Several columns are counted one
//! column at a time, in one map of at most `stop` entries at a time plus one 8-byte number per
//! row of the frame (`count_keys`), so the number of columns does not multiply the maps. Values
//! are keyed as the pivot groups them ([`Key`]).
//!
//! A column of a type that has no key here (lists, arrays, structs, objects) and a partitioned
//! column are counted by polars instead, exactly and without the bound; there, only the dataset
//! quotas and the refusals `check_pivot` makes before counting (of the rows of the frame, and of
//! the values its pivot would group) bound what polars reads.

use std::fmt;
use std::hash::Hash;

use p2a_core::polars::prelude::{
    BinaryChunked, BooleanChunked, Column, DataFrame, DataType, Float32Chunked, Float64Chunked,
    Int8Chunked, Int16Chunked, Int32Chunked, Int64Chunked, Int128Chunked, PlHashMap, PlHashSet,
    PolarsResult, StringChunked, UInt8Chunked, UInt16Chunked, UInt32Chunked, UInt64Chunked,
    UInt128Chunked,
};

/// A number counted up to a bound: exact when `exact`, a lower bound otherwise. It shows as the
/// number, or as `at least` the number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Count {
    pub(super) n: u64,
    pub(super) exact: bool,
}

impl Count {
    /// An exact count.
    pub(super) fn exact(n: u64) -> Count {
        Count { n, exact: true }
    }

    /// At least `n`.
    pub(super) fn at_least(n: u64) -> Count {
        Count { n, exact: false }
    }

    /// `self + k`; a sum past `u64::MAX` is at least `u64::MAX`.
    pub(super) fn plus(self, k: u64) -> Count {
        match self.n.checked_add(k) {
            Some(n) => Count { n, ..self },
            None => Count::at_least(u64::MAX),
        }
    }

    /// `self * other`, exact when both are; a product past `u64::MAX` is at least `u64::MAX`.
    pub(super) fn times(self, other: Count) -> Count {
        match self.n.checked_mul(other.n) {
            Some(n) => Count {
                n,
                exact: self.exact && other.exact,
            },
            None => Count::at_least(u64::MAX),
        }
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exact {
            write!(f, "{}", self.n)
        } else {
            write!(f, "at least {}", self.n)
        }
    }
}

/// The distinct combinations of the values of the `names` columns of `df` (at least one, each
/// of which `df` has), counted up to `stop`: exact below `stop`, and `stop` as a lower bound
/// once that many have been seen. A null is one value, as in the pivot. Errors are polars'
/// (a column `df` lacks, or a type polars cannot group).
pub(super) fn count(df: &DataFrame, names: &[&str], stop: u64) -> PolarsResult<Count> {
    let columns = names
        .iter()
        .map(|name| df.column(name))
        .collect::<PolarsResult<Vec<_>>>()?;
    let height = df.height();
    let mut keyed = Vec::with_capacity(columns.len());
    for column in &columns {
        // A column yields one key per row of its own length (a scalar column carries its
        // length), and every column of a frame has the frame's height, so `count_keys` reads
        // `height` keys from each. A column of another length, which a frame cannot hold, is
        // left to polars rather than miscounted.
        match Keyed::of(column)? {
            Some(k) if column.len() == height => keyed.push(k),
            _ => return polars_count(df, names, &columns).map(Count::exact),
        }
    }
    let n = count_keys(&keyed, height, stop);
    Ok(Count { n, exact: n < stop })
}

/// polars' exact count, for columns [`Keyed::of`] has no keys for: `n_unique` of one column, the
/// groups of several.
fn polars_count(df: &DataFrame, names: &[&str], columns: &[&Column]) -> PolarsResult<u64> {
    match columns {
        [column] => Ok(column.n_unique()? as u64),
        _ => Ok(df.group_by(names.iter().copied())?.get_groups().len() as u64),
    }
}

/// A value as polars' pivot and group-by tell values apart: integers, and the dates, times,
/// durations, decimals and categories stored as integers, by their bits; floats by their bits
/// after making `-0.0` equal to `0.0` and every NaN one value; booleans as 0 and 1; strings and
/// binaries by their bytes; a null as one value of its own. The keys of one column all have one
/// variant besides [`Key::Null`], so values of different types never meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Key<'a> {
    Null,
    Bits(u64),
    Wide([u64; 2]),
    Bytes(&'a [u8]),
}

/// The bits a float is keyed by: `-0.0` as `0.0`, and every NaN as one NaN. An `f32` is
/// widened first, which keeps distinct values distinct.
fn float_bits(x: f64) -> u64 {
    if x.is_nan() {
        f64::NAN.to_bits()
    } else if x == 0.0 {
        0.0f64.to_bits()
    } else {
        x.to_bits()
    }
}

/// The halves of a 128-bit integer's bits.
fn wide(bits: u128) -> Key<'static> {
    Key::Wide([bits as u64, (bits >> 64) as u64])
}

/// A column's values in the physical form [`Key`]s are read from. The arrays are the dataset's
/// own: cloning a chunked array clones reference counts, not values.
enum Keyed {
    /// One value in each of `len` rows: a scalar column, or a column of the Null type.
    Constant(usize),
    Bool(BooleanChunked),
    I8(Int8Chunked),
    I16(Int16Chunked),
    I32(Int32Chunked),
    I64(Int64Chunked),
    I128(Int128Chunked),
    U8(UInt8Chunked),
    U16(UInt16Chunked),
    U32(UInt32Chunked),
    U64(UInt64Chunked),
    U128(UInt128Chunked),
    F32(Float32Chunked),
    F64(Float64Chunked),
    Str(StringChunked),
    Bin(BinaryChunked),
}

impl Keyed {
    /// `None` for a partitioned column, and for a type without keys (lists, arrays, structs,
    /// objects), which polars counts instead.
    fn of(column: &Column) -> PolarsResult<Option<Keyed>> {
        if let Some(scalar) = column.as_scalar_column() {
            return Ok(Some(Keyed::Constant(scalar.len())));
        }
        let Some(series) = column.as_series() else {
            return Ok(None);
        };
        let s = series.to_physical_repr();
        Ok(Some(match s.dtype() {
            DataType::Null => Keyed::Constant(s.len()),
            DataType::Boolean => Keyed::Bool(s.bool()?.clone()),
            DataType::Int8 => Keyed::I8(s.i8()?.clone()),
            DataType::Int16 => Keyed::I16(s.i16()?.clone()),
            DataType::Int32 => Keyed::I32(s.i32()?.clone()),
            DataType::Int64 => Keyed::I64(s.i64()?.clone()),
            DataType::Int128 => Keyed::I128(s.i128()?.clone()),
            DataType::UInt8 => Keyed::U8(s.u8()?.clone()),
            DataType::UInt16 => Keyed::U16(s.u16()?.clone()),
            DataType::UInt32 => Keyed::U32(s.u32()?.clone()),
            DataType::UInt64 => Keyed::U64(s.u64()?.clone()),
            DataType::UInt128 => Keyed::U128(s.u128()?.clone()),
            DataType::Float32 => Keyed::F32(s.f32()?.clone()),
            DataType::Float64 => Keyed::F64(s.f64()?.clone()),
            DataType::String => Keyed::Str(s.str()?.clone()),
            DataType::Binary => Keyed::Bin(s.binary()?.clone()),
            _ => return Ok(None),
        }))
    }

    /// The column's keys, row by row.
    fn keys(&self) -> Box<dyn Iterator<Item = Key<'_>> + '_> {
        /// Keys of the integer values of `$ca`, each converted to `u64` by `$bits`.
        macro_rules! bits {
            ($ca:expr, $bits:expr) => {
                Box::new(
                    $ca.iter()
                        .map(|v| v.map_or(Key::Null, |v| Key::Bits($bits(v)))),
                )
            };
        }
        match self {
            Keyed::Constant(len) => Box::new(std::iter::repeat_n(Key::Null, *len)),
            Keyed::Bool(ca) => bits!(ca, u64::from),
            // Signed values are sign-extended, which keeps them distinct within a column.
            Keyed::I8(ca) => bits!(ca, |v: i8| i64::from(v) as u64),
            Keyed::I16(ca) => bits!(ca, |v: i16| i64::from(v) as u64),
            Keyed::I32(ca) => bits!(ca, |v: i32| i64::from(v) as u64),
            Keyed::I64(ca) => bits!(ca, |v: i64| v as u64),
            Keyed::U8(ca) => bits!(ca, u64::from),
            Keyed::U16(ca) => bits!(ca, u64::from),
            Keyed::U32(ca) => bits!(ca, u64::from),
            Keyed::U64(ca) => bits!(ca, |v: u64| v),
            Keyed::F32(ca) => bits!(ca, |v: f32| float_bits(f64::from(v))),
            Keyed::F64(ca) => bits!(ca, float_bits),
            Keyed::I128(ca) => {
                Box::new(ca.iter().map(|v| v.map_or(Key::Null, |v| wide(v as u128))))
            }
            Keyed::U128(ca) => Box::new(ca.iter().map(|v| v.map_or(Key::Null, wide))),
            Keyed::Str(ca) => Box::new(
                ca.iter()
                    .map(|v| v.map_or(Key::Null, |v| Key::Bytes(v.as_bytes()))),
            ),
            Keyed::Bin(ca) => Box::new(ca.iter().map(|v| v.map_or(Key::Null, Key::Bytes))),
        }
    }
}

/// The number `ids` gives `key`, a new one (the next) if it has none.
fn number<K: Hash + Eq>(ids: &mut PlHashMap<K, u64>, key: K) -> u64 {
    let next = ids.len() as u64;
    *ids.entry(key).or_insert(next)
}

/// The distinct combinations of the keys of `columns`, each of which has exactly `height` keys
/// ([`count`] checks it), counted up to `stop` (returned once that many are seen; 0 without
/// columns).
///
/// One column's keys go into one set. Several are numbered one column at a time: each row
/// carries the number of its combination of the columns read so far, and the next column numbers
/// the distinct pairs (that number, the row's key in the column) in a map that is dropped once
/// the column is read. Fewer columns never have more distinct combinations than all of them, so
/// the count stops as soon as any map reaches `stop`. It holds one map of at most `stop` entries
/// at a time, plus one number per row, however many columns there are.
fn count_keys(columns: &[Keyed], height: usize, stop: u64) -> u64 {
    if columns.is_empty() || stop == 0 {
        return 0;
    }
    let stop_len = usize::try_from(stop).unwrap_or(usize::MAX);
    if let [column] = columns {
        let mut seen = PlHashSet::default();
        for key in column.keys().take(height) {
            seen.insert(key);
            if seen.len() >= stop_len {
                return stop;
            }
        }
        return seen.len() as u64;
    }
    // Every row starts in combination 0 of no columns.
    let mut ids = vec![0u64; height];
    let mut distinct = 0;
    for column in columns {
        let mut pairs: PlHashMap<(u64, Key<'_>), u64> = PlHashMap::default();
        // The column has one key for each of the `height` rows, so each row's number moves on.
        for (id, key) in ids.iter_mut().zip(column.keys()) {
            *id = number(&mut pairs, (*id, key));
            if pairs.len() >= stop_len {
                return stop;
            }
        }
        distinct = pairs.len();
    }
    distinct as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_core::polars::prelude::{
        AnyValue, Categories, IntoColumn, NamedFrom, Scalar, Series, TimeUnit, df,
    };

    /// polars' count of the groups of `names`, which the pivot's rows and columns follow.
    fn groups(df: &DataFrame, names: &[&str]) -> u64 {
        df.group_by(names.iter().copied())
            .unwrap()
            .get_groups()
            .len() as u64
    }

    fn unbounded(df: &DataFrame, names: &[&str]) -> u64 {
        let c = count(df, names, u64::MAX).unwrap();
        assert!(c.exact, "{c:?}");
        c.n
    }

    #[test]
    fn the_count_is_exact_at_and_just_over_the_bound_and_a_lower_bound_past_it() {
        // Four distinct values, one of them null.
        let df = df!("k" => [Some("a"), Some("b"), None, Some("c"), Some("a"), None, Some("b")])
            .unwrap();
        assert_eq!(groups(&df, &["k"]), 4);
        // A bound of 3 that a count of 3 passes: the count stops at 4, its exact value.
        assert_eq!(count(&df, &["k"], 5).unwrap(), Count::exact(4));
        assert_eq!(count(&df, &["k"], 4).unwrap(), Count::at_least(4));
        assert_eq!(count(&df, &["k"], 3).unwrap(), Count::at_least(3));
        assert_eq!(count(&df, &["k"], 1).unwrap(), Count::at_least(1));
        // The same over a numeric column with a null, and over a combination of the two.
        let df = df!(
            "k" => [Some(1.5), Some(2.5), None, Some(1.5), Some(3.5), None],
            "j" => [Some(1i64), Some(1), Some(1), None, Some(1), Some(2)]
        )
        .unwrap();
        assert_eq!(groups(&df, &["k"]), 4);
        assert_eq!(count(&df, &["k"], 5).unwrap(), Count::exact(4));
        assert_eq!(count(&df, &["k"], 4).unwrap(), Count::at_least(4));
        // (1.5,1) (2.5,1) (null,1) (1.5,null) (3.5,1) (null,2)
        assert_eq!(groups(&df, &["k", "j"]), 6);
        assert_eq!(count(&df, &["k", "j"], 7).unwrap(), Count::exact(6));
        assert_eq!(count(&df, &["k", "j"], 6).unwrap(), Count::at_least(6));
        assert_eq!(count(&df, &["k", "j"], 2).unwrap(), Count::at_least(2));
    }

    #[test]
    fn values_are_told_apart_as_polars_groups_them() {
        let columns: Vec<Column> = vec![
            Series::new("bool".into(), [Some(true), Some(false), None, Some(true)]).into(),
            Series::new("i8".into(), [Some(-1i8), Some(1), None, Some(-1)]).into(),
            Series::new("i16".into(), [Some(-1i16), Some(i16::MAX), None, Some(-1)]).into(),
            Series::new("i32".into(), [Some(-1i32), Some(7), None, Some(7)]).into(),
            Series::new("i64".into(), [Some(i64::MIN), Some(-1), None, Some(-1)]).into(),
            Series::new(
                "i128".into(),
                [Some(-1i128), Some(1 << 100), None, Some(-1)],
            )
            .into(),
            Series::new("u8".into(), [Some(255u8), Some(1), None, Some(1)]).into(),
            Series::new("u16".into(), [Some(9u16), Some(1), None, Some(9)]).into(),
            Series::new("u32".into(), [Some(u32::MAX), Some(1), None, Some(1)]).into(),
            Series::new("u64".into(), [Some(u64::MAX), Some(1), None, Some(1)]).into(),
            Series::new("u128".into(), [Some(u128::MAX), Some(1), None, Some(1)]).into(),
            // -0.0 is 0.0, and every NaN is one value.
            Series::new(
                "f32".into(),
                [
                    Some(0.0f32),
                    Some(-0.0),
                    Some(f32::NAN),
                    Some(-f32::NAN),
                    None,
                    Some(0.5),
                ],
            )
            .into(),
            Series::new(
                "f64".into(),
                [
                    Some(0.0f64),
                    Some(-0.0),
                    Some(f64::NAN),
                    Some(-f64::NAN),
                    None,
                    Some(0.5),
                ],
            )
            .into(),
            Series::new("str".into(), [Some("a"), Some(""), None, Some("a")]).into(),
            Series::new(
                "bin".into(),
                [Some(&b"a"[..]), Some(b"\0"), None, Some(b"a")],
            )
            .into(),
            Series::new_null("null".into(), 3).into(),
            Series::new("date".into(), [Some(3i32), Some(4), None, Some(3)])
                .cast(&DataType::Date)
                .unwrap()
                .into(),
            Series::new("datetime".into(), [Some(3i64), Some(4), None, Some(4)])
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()
                .into(),
            Series::new("cat".into(), [Some("x"), Some("y"), None, Some("x")])
                .cast(&DataType::from_categories(Categories::global()))
                .unwrap()
                .into(),
            Column::new_scalar(
                "scalar".into(),
                Scalar::new(DataType::Int64, AnyValue::Int64(3)),
                5,
            ),
            Column::new_scalar(
                "empty".into(),
                Scalar::new(DataType::Int64, AnyValue::Int64(3)),
                0,
            ),
            Column::new_scalar("nulls".into(), Scalar::null(DataType::String), 4),
            // No keys: counted by polars.
            Series::new(
                "list".into(),
                [
                    Series::new("".into(), [1i64, 2]),
                    Series::new("".into(), [1i64]),
                    Series::new("".into(), [1i64, 2]),
                ],
            )
            .into(),
        ];
        let expected = [
            ("bool", 3),
            ("i8", 3),
            ("i16", 3),
            ("i32", 3),
            ("i64", 3),
            ("i128", 3),
            ("u8", 3),
            ("u16", 3),
            ("u32", 3),
            ("u64", 3),
            ("u128", 3),
            ("f32", 4),
            ("f64", 4),
            ("str", 3),
            ("bin", 3),
            ("null", 1),
            ("date", 3),
            ("datetime", 3),
            ("cat", 3),
            ("scalar", 1),
            ("empty", 0),
            ("nulls", 1),
            ("list", 2),
        ];
        assert_eq!(columns.len(), expected.len());
        for (column, (name, n)) in columns.into_iter().zip(expected) {
            assert_eq!(column.name().as_str(), name);
            let df = DataFrame::new(vec![column]).unwrap();
            assert_eq!(groups(&df, &[name]), n, "{name}: polars");
            assert_eq!(unbounded(&df, &[name]), n, "{name}");
        }
    }

    #[test]
    fn combinations_are_counted_across_columns() {
        // `b` is determined by `a` except in the last two rows; nulls are values of their own.
        let df = df!(
            "a" => [Some(1i64), Some(1), Some(2), Some(2), None, None, Some(1), Some(3)],
            "b" => [Some("x"), Some("x"), Some("y"), Some("y"), None, None, Some("y"), None],
            "c" => [0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 0.5, 0.5]
        )
        .unwrap();
        for names in [
            &["a"][..],
            &["a", "b"],
            &["b", "a"],
            &["a", "c"],
            &["a", "b", "c"],
            &["c", "b"],
        ] {
            let n = groups(&df, names);
            assert_eq!(unbounded(&df, names), n, "{names:?}");
            assert_eq!(count(&df, names, n + 1).unwrap(), Count::exact(n));
            assert_eq!(count(&df, names, n).unwrap(), Count::at_least(n));
        }
        // A scalar column is one value in every row and leaves the combinations as they were.
        let mut with_scalar = df.clone();
        with_scalar
            .with_column(Column::new_scalar(
                "s".into(),
                Scalar::new(DataType::String, AnyValue::StringOwned("k".into())),
                df.height(),
            ))
            .unwrap();
        assert_eq!(
            unbounded(&with_scalar, &["a", "s", "b"]),
            groups(&df, &["a", "b"])
        );
        // A count over several columns stops at the first column whose combinations reach the
        // bound: `a` alone has 4 values, so the pairs with `c` never need counting.
        assert_eq!(groups(&df, &["a"]), 4);
        assert_eq!(count(&df, &["a", "c"], 4).unwrap(), Count::at_least(4));
        assert_eq!(count(&df, &["a", "c"], 3).unwrap(), Count::at_least(3));
        // Several columns, one without keys: polars counts them.
        let mut with_list = df.clone();
        let lists: Vec<Series> = (0..df.height())
            .map(|i| Series::new("".into(), [(i % 2) as i64]))
            .collect();
        with_list
            .with_column(Series::new("l".into(), lists).into_column())
            .unwrap();
        assert_eq!(
            unbounded(&with_list, &["a", "l"]),
            groups(&with_list, &["a", "l"])
        );
    }

    #[test]
    fn counts_show_whether_they_are_exact() {
        assert_eq!(Count::exact(5).to_string(), "5");
        assert_eq!(Count::at_least(5).to_string(), "at least 5");
        assert_eq!(Count::exact(5).plus(2), Count::exact(7));
        assert_eq!(Count::at_least(5).plus(2), Count::at_least(7));
        assert_eq!(Count::exact(u64::MAX).plus(1), Count::at_least(u64::MAX));
        assert_eq!(Count::exact(3).times(Count::exact(4)), Count::exact(12));
        assert_eq!(
            Count::exact(3).times(Count::at_least(4)),
            Count::at_least(12)
        );
        assert_eq!(
            Count::exact(u64::MAX).times(Count::exact(2)),
            Count::at_least(u64::MAX)
        );
    }
}
