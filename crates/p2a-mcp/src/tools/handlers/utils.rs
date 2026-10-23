//! Utility tool handlers (seed management, reports, session export/import).
//!
//! This module defines utility tools using the `#[tool_router(router = utils_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::path_jail;
use crate::server::AnalyticsServer;
use crate::tools::limits::{Family, LimitViolation};
use crate::tools::requests::utils::{
    ColumnSpecInput, ExportSessionRequest, GenerateRandomDataRequest, GenerateReportRequest,
    GetSeedRequest, ImportSessionRequest, ReportContentInput, ReportSectionInput,
    ServerStatsRequest, SetSeedRequest,
};

use p2a_core::{
    Dataset, HtmlReport, ReportSection, ReportTable,
    polars::prelude::{AnyValue, Column, DataFrame},
    simulation::{ColumnSpec, Distribution, generate_random_data},
};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use std::collections::HashMap;

/// A writer that refuses to pass more than `cap` bytes to `inner`: the shared
/// byte cap for exports. A write that would take the total past the cap is
/// refused whole (nothing of it reaches `inner`), marks the writer as
/// exceeded and fails with an I/O error; `None` means no cap.
pub(crate) struct CappedWriter<W> {
    inner: W,
    cap: Option<u64>,
    written: u64,
    exceeded: bool,
}

impl<W> CappedWriter<W> {
    pub(crate) fn new(inner: W, cap: Option<u64>) -> Self {
        Self {
            inner,
            cap,
            written: 0,
            exceeded: false,
        }
    }

    /// Whether a write was refused because of the cap.
    pub(crate) fn exceeded(&self) -> bool {
        self.exceeded
    }

    /// Bytes passed to the inner writer so far.
    pub(crate) fn written(&self) -> u64 {
        self.written
    }

    pub(crate) fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: std::io::Write> std::io::Write for CappedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(cap) = self.cap
            && self.written.saturating_add(buf.len() as u64) > cap
        {
            self.exceeded = true;
            return Err(std::io::Error::other("export size limit reached"));
        }
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Write the `export_session` JSON for `datasets` to `w` in one streaming
/// pass (compact, no intermediate `Value` tree): top-level keys `version`,
/// `created_at`, `datasets`; datasets in name order, each with `schema`,
/// `n_rows`, `n_cols` and, with `include_data`, `data` (one array per column,
/// in column order). Cell values map as the previous tree builder mapped them.
fn write_session_json<W: std::io::Write>(
    datasets: &HashMap<String, Dataset>,
    include_data: bool,
    created_at: &str,
    w: W,
) -> serde_json::Result<()> {
    serde_json::to_writer(
        w,
        &SessionView {
            datasets,
            include_data,
            created_at,
        },
    )
}

struct SessionView<'a> {
    datasets: &'a HashMap<String, Dataset>,
    include_data: bool,
    created_at: &'a str,
}

impl Serialize for SessionView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(3))?;
        map.serialize_entry("version", "1.0")?;
        map.serialize_entry("created_at", self.created_at)?;
        map.serialize_entry(
            "datasets",
            &DatasetsView {
                datasets: self.datasets,
                include_data: self.include_data,
            },
        )?;
        map.end()
    }
}

struct DatasetsView<'a> {
    datasets: &'a HashMap<String, Dataset>,
    include_data: bool,
}

impl Serialize for DatasetsView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut names: Vec<&String> = self.datasets.keys().collect();
        names.sort();
        let mut map = s.serialize_map(Some(names.len()))?;
        for name in names {
            let view = DatasetView {
                df: self.datasets[name].df(),
                include_data: self.include_data,
            };
            map.serialize_entry(name, &view)?;
        }
        map.end()
    }
}

struct DatasetView<'a> {
    df: &'a DataFrame,
    include_data: bool,
}

impl Serialize for DatasetView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(if self.include_data { 4 } else { 3 }))?;
        map.serialize_entry("schema", &SchemaView(self.df.get_columns()))?;
        map.serialize_entry("n_rows", &self.df.height())?;
        map.serialize_entry("n_cols", &self.df.width())?;
        if self.include_data {
            map.serialize_entry("data", &ColumnsView(self.df.get_columns()))?;
        }
        map.end()
    }
}

struct SchemaView<'a>(&'a [Column]);

impl Serialize for SchemaView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for col in self.0 {
            seq.serialize_element(&SchemaEntry(col))?;
        }
        seq.end()
    }
}

struct SchemaEntry<'a>(&'a Column);

impl Serialize for SchemaEntry<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(2))?;
        map.serialize_entry("name", self.0.name().as_str())?;
        map.serialize_entry("dtype", &format!("{:?}", self.0.dtype()))?;
        map.end()
    }
}

struct ColumnsView<'a>(&'a [Column]);

impl Serialize for ColumnsView<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error as _;
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for col in self.0 {
            p2a_core::resource::checkpoint().map_err(S::Error::custom)?;
            map.serialize_entry(col.name().as_str(), &ColumnValues(col))?;
        }
        map.end()
    }
}

struct ColumnValues<'a>(&'a Column);

impl Serialize for ColumnValues<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let col = self.0;
        let mut seq = s.serialize_seq(Some(col.len()))?;
        for i in 0..col.len() {
            seq.serialize_element(&SessionCell(col.get(i).ok()))?;
        }
        seq.end()
    }
}

/// One cell of a session export; `None` (a failed read) is null.
struct SessionCell<'a>(Option<AnyValue<'a>>);

impl Serialize for SessionCell<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.0 {
            None | Some(AnyValue::Null) => s.serialize_none(),
            Some(AnyValue::Boolean(b)) => s.serialize_bool(*b),
            Some(AnyValue::Int8(v)) => s.serialize_i64(i64::from(*v)),
            Some(AnyValue::Int16(v)) => s.serialize_i64(i64::from(*v)),
            Some(AnyValue::Int32(v)) => s.serialize_i64(i64::from(*v)),
            Some(AnyValue::Int64(v)) => s.serialize_i64(*v),
            Some(AnyValue::UInt8(v)) => s.serialize_u64(u64::from(*v)),
            Some(AnyValue::UInt16(v)) => s.serialize_u64(u64::from(*v)),
            Some(AnyValue::UInt32(v)) => s.serialize_u64(u64::from(*v)),
            Some(AnyValue::UInt64(v)) => s.serialize_u64(*v),
            // The previous builder widened f32 to f64 (`json!`); non-finite
            // values are written as null.
            Some(AnyValue::Float32(v)) => s.serialize_f64(f64::from(*v)),
            Some(AnyValue::Float64(v)) => s.serialize_f64(*v),
            Some(AnyValue::String(v)) => s.serialize_str(v),
            Some(other) => s.serialize_str(&format!("{other:?}")),
        }
    }
}

// ============================================================================
// Streaming session import
// ============================================================================

/// One dataset of a session file, as `import_session` reads it.
#[derive(Debug)]
enum ParsedDataset {
    /// The dataset's value is not an object.
    Invalid,
    /// No `data` object (a metadata-only export).
    NoData,
    /// The `data` object's columns, in file order.
    Columns(Vec<(String, ParsedColumn)>),
}

/// One column of a session file. The type is decided by the first non-null
/// element, as the previous import decided it.
#[derive(Debug)]
enum ParsedColumn {
    F64(Vec<Option<f64>>),
    I64(Vec<Option<i64>>),
    Bool(Vec<Option<bool>>),
    Str(Vec<Option<String>>),
    /// The column's value is not an array (skipped).
    NotArray,
}

/// Shared state of one streaming parse: the bytes charged against the
/// budget, and why the parse was stopped (budget or cancellation), if it was.
struct ImportState {
    used: std::cell::Cell<u64>,
    budget: Option<u64>,
    stop: std::cell::RefCell<Option<String>>,
}

impl ImportState {
    /// Charge `bytes`; an error once the total passes the budget.
    fn charge<E: serde::de::Error>(&self, bytes: u64) -> Result<(), E> {
        let used = self.used.get().saturating_add(bytes);
        self.used.set(used);
        match self.budget {
            Some(budget) if used > budget => Err(self.stopped(format!(
                "the session file needs more than this server's import limit of {budget} bytes \
                 in memory"
            ))),
            _ => Ok(()),
        }
    }

    /// Give back `bytes` charged for something already freed.
    fn refund(&self, bytes: u64) {
        self.used.set(self.used.get().saturating_sub(bytes));
    }

    fn checkpoint<E: serde::de::Error>(&self) -> Result<(), E> {
        p2a_core::resource::checkpoint().map_err(|e| self.stopped(e.to_string()))
    }

    fn stopped<E: serde::de::Error>(&self, reason: String) -> E {
        let err = E::custom(&reason);
        *self.stop.borrow_mut() = Some(reason);
        err
    }
}

/// Parse a session file in one streaming pass, straight into typed column
/// vectors, without a `serde_json::Value` tree for the document. The bytes
/// charged (values, the text of string cells and names, map entries, and the
/// one element being read) stay within `budget`; allocator and container
/// growth slack (vector capacity, hash-map buckets) is not charged, so peak
/// memory is a small constant multiple of it.
fn parse_session_json(
    reader: impl std::io::Read,
    budget: Option<u64>,
) -> Result<Vec<(String, ParsedDataset)>, String> {
    use serde::de::DeserializeSeed;
    let state = ImportState {
        used: std::cell::Cell::new(0),
        budget,
        stop: std::cell::RefCell::new(None),
    };
    let mut de = serde_json::Deserializer::from_reader(reader);
    let parsed = TopSeed(&state)
        .deserialize(&mut de)
        .and_then(|datasets| de.end().map(|()| datasets));
    match parsed {
        Ok(Some(datasets)) => Ok(datasets),
        Ok(None) => Err("Invalid session file: missing 'datasets'".to_string()),
        Err(e) => Err(match state.stop.borrow_mut().take() {
            Some(reason) => reason,
            None if e.is_io() => format!("Failed to read the session file: {e}"),
            None => format!("Invalid session file: {e}"),
        }),
    }
}

/// Implements every non-map `Visitor` method by consuming the value and
/// returning `$fallback` (serde_json calls only these for JSON values).
macro_rules! non_map_values_are {
    ($fallback:expr) => {
        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok($fallback)
        }
        fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
            Ok($fallback)
        }
        fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
            Ok($fallback)
        }
        fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
            Ok($fallback)
        }
        fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
            Ok($fallback)
        }
        fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self::Value, E> {
            Ok($fallback)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut seq: A,
        ) -> Result<Self::Value, A::Error> {
            while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok($fallback)
        }
    };
}

/// A JSON object's entries in file order, where a later duplicate key
/// replaces the earlier value in place (the last duplicate wins, as in
/// `serde_json::Map`). The index makes each insert O(1), so a file with many
/// keys parses in linear time.
struct ObjectEntries<V> {
    entries: Vec<(String, V)>,
    index: HashMap<String, usize>,
}

impl<V> ObjectEntries<V> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Insert `(key, value)`. A new key is charged against `state` for its
    /// second copy (in the index) and both entries; the first copy was
    /// charged when the key was read.
    fn insert<E: serde::de::Error>(
        &mut self,
        key: String,
        value: V,
        state: &ImportState,
    ) -> Result<(), E> {
        if let Some(entry) = self.index.get(&key).and_then(|&i| self.entries.get_mut(i)) {
            entry.1 = value;
            return Ok(());
        }
        let entry_bytes =
            std::mem::size_of::<(String, V)>() + std::mem::size_of::<(String, usize)>();
        state.charge((key.len() + entry_bytes) as u64)?;
        self.index.insert(key.clone(), self.entries.len());
        self.entries.push((key, value));
        Ok(())
    }

    fn into_vec(self) -> Vec<(String, V)> {
        self.entries
    }
}

/// The document: `Some(datasets)`, or `None` when it is not an object or has
/// no `datasets` object. Other top-level keys are skipped.
struct TopSeed<'s>(&'s ImportState);

impl<'de> serde::de::DeserializeSeed<'de> for TopSeed<'_> {
    type Value = Option<Vec<(String, ParsedDataset)>>;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for TopSeed<'_> {
    type Value = Option<Vec<(String, ParsedDataset)>>;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a session object")
    }
    non_map_values_are!(None);
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut datasets = None;
        while let Some(key) = map.next_key::<String>()? {
            if key == "datasets" {
                datasets = map.next_value_seed(DatasetsSeed(self.0))?;
            } else {
                map.next_value::<serde::de::IgnoredAny>()?;
            }
        }
        Ok(datasets)
    }
}

/// The `datasets` value: `Some` for an object, in file order.
struct DatasetsSeed<'s>(&'s ImportState);

impl<'de> serde::de::DeserializeSeed<'de> for DatasetsSeed<'_> {
    type Value = Option<Vec<(String, ParsedDataset)>>;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for DatasetsSeed<'_> {
    type Value = Option<Vec<(String, ParsedDataset)>>;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an object of datasets")
    }
    non_map_values_are!(None);
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut out = ObjectEntries::new();
        while let Some(name) = map.next_key::<String>()? {
            self.0.checkpoint()?;
            self.0.charge(name.len() as u64)?;
            let dataset = map.next_value_seed(DatasetSeed(self.0))?;
            out.insert(name, dataset, self.0)?;
        }
        Ok(Some(out.into_vec()))
    }
}

/// One dataset's value: only its `data` key is read.
struct DatasetSeed<'s>(&'s ImportState);

impl<'de> serde::de::DeserializeSeed<'de> for DatasetSeed<'_> {
    type Value = ParsedDataset;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for DatasetSeed<'_> {
    type Value = ParsedDataset;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a dataset object")
    }
    non_map_values_are!(ParsedDataset::Invalid);
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut data = None;
        while let Some(key) = map.next_key::<String>()? {
            if key == "data" {
                data = map.next_value_seed(DataSeed(self.0))?;
            } else {
                map.next_value::<serde::de::IgnoredAny>()?;
            }
        }
        Ok(data.map_or(ParsedDataset::NoData, ParsedDataset::Columns))
    }
}

/// A dataset's `data` value: `Some(columns)` for an object.
struct DataSeed<'s>(&'s ImportState);

impl<'de> serde::de::DeserializeSeed<'de> for DataSeed<'_> {
    type Value = Option<Vec<(String, ParsedColumn)>>;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for DataSeed<'_> {
    type Value = Option<Vec<(String, ParsedColumn)>>;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an object of columns")
    }
    non_map_values_are!(None);
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut columns = ObjectEntries::new();
        while let Some(name) = map.next_key::<String>()? {
            self.0.checkpoint()?;
            self.0.charge(name.len() as u64)?;
            let column = map.next_value_seed(ColumnSeed(self.0))?;
            columns.insert(name, column, self.0)?;
        }
        Ok(Some(columns.into_vec()))
    }
}

/// One column's value: an array becomes a typed vector, anything else
/// [`ParsedColumn::NotArray`].
struct ColumnSeed<'s>(&'s ImportState);

impl<'de> serde::de::DeserializeSeed<'de> for ColumnSeed<'_> {
    type Value = ParsedColumn;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }
}

/// A column being filled: `Pending(n)` until the first non-null element has
/// decided the type (`n` leading nulls so far).
enum ColumnBuilder {
    Pending(usize),
    Typed(ParsedColumn),
}

impl ColumnBuilder {
    fn push<E: serde::de::Error>(
        &mut self,
        v: serde_json::Value,
        state: &ImportState,
    ) -> Result<(), E> {
        use serde_json::Value;
        if let ColumnBuilder::Pending(nulls) = self {
            if v.is_null() {
                *nulls += 1;
                return Ok(());
            }
            let leading = *nulls;
            let column = match &v {
                Value::Number(n) if n.is_f64() => {
                    ParsedColumn::F64(pending_nulls::<f64, E>(leading, state)?)
                }
                Value::Number(_) => ParsedColumn::I64(pending_nulls::<i64, E>(leading, state)?),
                Value::Bool(_) => ParsedColumn::Bool(pending_nulls::<bool, E>(leading, state)?),
                _ => ParsedColumn::Str(pending_nulls::<String, E>(leading, state)?),
            };
            *self = ColumnBuilder::Typed(column);
        }
        let ColumnBuilder::Typed(column) = self else {
            unreachable!("the type is decided above");
        };
        match column {
            ParsedColumn::F64(vals) => {
                state.charge(std::mem::size_of::<Option<f64>>() as u64)?;
                vals.push(v.as_f64());
            }
            ParsedColumn::I64(vals) => {
                state.charge(std::mem::size_of::<Option<i64>>() as u64)?;
                vals.push(v.as_i64());
            }
            ParsedColumn::Bool(vals) => {
                state.charge(std::mem::size_of::<Option<bool>>() as u64)?;
                vals.push(v.as_bool());
            }
            ParsedColumn::Str(vals) => {
                let s = match v {
                    Value::Null => None,
                    Value::String(s) => Some(s),
                    other => Some(other.to_string()),
                };
                let text = s.as_ref().map_or(0, String::len) as u64;
                state.charge(std::mem::size_of::<Option<String>>() as u64 + text)?;
                vals.push(s);
            }
            ParsedColumn::NotArray => {}
        }
        Ok(())
    }

    fn finish<E: serde::de::Error>(self, state: &ImportState) -> Result<ParsedColumn, E> {
        Ok(match self {
            // An all-null (or empty) column is a string column, as before.
            ColumnBuilder::Pending(nulls) => {
                ParsedColumn::Str(pending_nulls::<String, E>(nulls, state)?)
            }
            ColumnBuilder::Typed(column) => column,
        })
    }
}

/// `n` leading `None`s for a column whose type has just been decided.
fn pending_nulls<T, E: serde::de::Error>(
    n: usize,
    state: &ImportState,
) -> Result<Vec<Option<T>>, E> {
    let size = std::mem::size_of::<Option<T>>() as u64;
    state.charge((n as u64).saturating_mul(size))?;
    let mut v = Vec::new();
    v.resize_with(n, || None);
    Ok(v)
}

impl<'de> serde::de::Visitor<'de> for ColumnSeed<'_> {
    type Value = ParsedColumn;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an array of values")
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(ParsedColumn::NotArray)
    }
    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
        Ok(ParsedColumn::NotArray)
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
        Ok(ParsedColumn::NotArray)
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
        Ok(ParsedColumn::NotArray)
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
        Ok(ParsedColumn::NotArray)
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<Self::Value, E> {
        Ok(ParsedColumn::NotArray)
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        while map
            .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
            .is_some()
        {}
        Ok(ParsedColumn::NotArray)
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let state = self.0;
        let mut column = ColumnBuilder::Pending(0);
        loop {
            // The element is held as one transient value, charged while it is
            // read and given back once it has been converted.
            let before = state.used.get();
            let Some(v) = seq.next_element_seed(ElementSeed(state))? else {
                break;
            };
            state.refund(state.used.get().saturating_sub(before));
            column.push(v, state)?;
        }
        column.finish(state)
    }
}

/// One element of a column, read as a `serde_json::Value` whose nodes and
/// text are charged against the budget as they are read.
struct ElementSeed<'s>(&'s ImportState);

impl<'de> serde::de::DeserializeSeed<'de> for ElementSeed<'_> {
    type Value = serde_json::Value;
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Self::Value, D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> serde::de::Visitor<'de> for ElementSeed<'_> {
    type Value = serde_json::Value;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Null)
    }
    fn visit_bool<E: serde::de::Error>(self, b: bool) -> Result<Self::Value, E> {
        Ok(serde_json::Value::Bool(b))
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(v.into())
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(v.into())
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Ok(serde_json::Number::from_f64(v).map_or(serde_json::Value::Null, Into::into))
    }
    fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<Self::Value, E> {
        self.0.charge(s.len() as u64)?;
        Ok(serde_json::Value::String(s.to_string()))
    }
    fn visit_string<E: serde::de::Error>(self, s: String) -> Result<Self::Value, E> {
        self.0.charge(s.len() as u64)?;
        Ok(serde_json::Value::String(s))
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element_seed(ElementSeed(self.0))? {
            self.0
                .charge(std::mem::size_of::<serde_json::Value>() as u64)?;
            items.push(item);
        }
        Ok(serde_json::Value::Array(items))
    }
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut object = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            self.0
                .charge(key.len() as u64 + std::mem::size_of::<serde_json::Value>() as u64 * 2)?;
            let value = map.next_value_seed(ElementSeed(self.0))?;
            object.insert(key, value);
        }
        Ok(serde_json::Value::Object(object))
    }
}

/// A parsed dataset as a frame, with the per-dataset error texts of the
/// previous import (`invalid format`, `no data field (metadata-only
/// session)`, `no column data found`, `DataFrame error - ...`). Columns that
/// were not arrays are skipped.
fn build_dataset(parsed: ParsedDataset) -> Result<DataFrame, String> {
    use p2a_core::polars::prelude::{NamedFrom, Series};
    let columns = match parsed {
        ParsedDataset::Invalid => return Err("invalid format".to_string()),
        ParsedDataset::NoData => return Err("no data field (metadata-only session)".to_string()),
        ParsedDataset::Columns(columns) => columns,
    };
    let columns: Vec<Column> = columns
        .into_iter()
        .filter_map(|(name, column)| {
            let series = match column {
                ParsedColumn::F64(v) => Series::new(name.as_str().into(), v),
                ParsedColumn::I64(v) => Series::new(name.as_str().into(), v),
                ParsedColumn::Bool(v) => Series::new(name.as_str().into(), v),
                ParsedColumn::Str(v) => Series::new(name.as_str().into(), v),
                ParsedColumn::NotArray => return None,
            };
            Some(series.into())
        })
        .collect();
    if columns.is_empty() {
        return Err("no column data found".to_string());
    }
    DataFrame::new(columns).map_err(|e| format!("DataFrame error - {}", e))
}

/// A refusal by a named budget, as an in-band tool error (logged at `info`).
fn budget_refusal(v: LimitViolation) -> CallToolResult {
    tracing::info!(
        tool = %v.tool,
        override_key = v.override_key,
        requested = %v.requested,
        limit = %v.limit,
        "request refused by a named budget"
    );
    CallToolResult::error(vec![Content::text(v.message)])
}

/// A refusal by a family cap (string length, array length), as an in-band
/// tool error (logged at `info`).
fn cap_refusal(
    tool: &str,
    knob: &'static str,
    requested: usize,
    limit: u64,
    message: String,
) -> CallToolResult {
    tracing::info!(
        tool,
        override_key = knob,
        requested,
        limit,
        "request refused by a family cap"
    );
    CallToolResult::error(vec![Content::text(message)])
}

/// Bytes one row of a generated column costs once materialised (exports,
/// `describe`): 8 for every numeric distribution, a 16-byte view plus the
/// text for string columns.
fn generated_row_width(dist: &Distribution) -> u64 {
    match dist {
        Distribution::ConstantString { value } => 16u64.saturating_add(value.len() as u64),
        Distribution::Categorical { categories, .. } => {
            let longest = categories.iter().map(String::len).max().unwrap_or(0);
            16u64.saturating_add(longest as u64)
        }
        _ => 8,
    }
}

/// The hosted per-string and per-array caps for one generated column.
fn check_generated_column(
    name: &str,
    dist: &Distribution,
    string_cap: Option<u64>,
    array_cap: Option<u64>,
) -> Result<(), CallToolResult> {
    const TOOL: &str = "generate_random_data";
    if let Some(cap) = string_cap {
        let mut strings: Vec<&str> = vec![name];
        match dist {
            Distribution::ConstantString { value } => strings.push(value),
            Distribution::Categorical { categories, .. } => {
                strings.extend(categories.iter().map(String::as_str))
            }
            _ => {}
        }
        if let Some(long) = strings.iter().find(|s| s.len() as u64 > cap) {
            return Err(cap_refusal(
                TOOL,
                "string_bytes",
                long.len(),
                cap,
                format!(
                    "Column `{}`: a string of {} bytes exceeds this server's limit of {cap} \
                     bytes per string. Use shorter text.",
                    shown_name(name),
                    long.len()
                ),
            ));
        }
    }
    if let (Some(cap), Distribution::Categorical { categories, .. }) = (array_cap, dist) {
        let k = categories.len();
        if k as u64 > cap {
            return Err(cap_refusal(
                TOOL,
                "array_length",
                k,
                cap,
                format!(
                    "Column `{}`: {k} categories exceed this server's limit of {cap}. \
                     Use fewer categories.",
                    shown_name(name)
                ),
            ));
        }
    }
    Ok(())
}

/// A column name for a message: at most 64 characters, cut at a char boundary.
fn shown_name(name: &str) -> String {
    match name.char_indices().nth(64) {
        Some((cut, _)) => format!("{}…", &name[..cut]),
        None => name.to_string(),
    }
}

#[tool_router(router = utils_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Seed Management Tools
    // ========================================================================

    /// Set the global random seed for ML reproducibility.
    #[tool(
        description = "Set a global random seed for ML operations (kmeans, random_forest, tsne). When set, ML tools will use this seed as a fallback if no per-tool seed is specified. Clear by calling with no seed value."
    )]
    pub async fn set_seed(
        &self,
        Parameters(request): Parameters<SetSeedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut global_seed = self.global_seed.write().await;
        *global_seed = request.seed;

        match request.seed {
            Some(seed) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Global random seed set to: {}\n\
                 This seed will be used by ML tools (kmeans, random_forest, tsne) unless overridden per-tool.",
                seed
            ))])),
            None => Ok(CallToolResult::success(vec![Content::text(
                "Global random seed cleared. ML tools will use random initialization unless a per-tool seed is specified.".to_string()
            )])),
        }
    }

    /// Get the current global random seed.
    #[tool(
        description = "Get the current global random seed setting and list which ML tools support seeded reproducibility."
    )]
    pub async fn get_seed(
        &self,
        Parameters(_request): Parameters<GetSeedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let global_seed = self.global_seed.read().await;

        let seed_status = match *global_seed {
            Some(seed) => format!("Current global seed: {}", seed),
            None => "No global seed set (using random initialization)".to_string(),
        };

        let output = format!(
            "Seed Management\n{}\n\
             {}\n\n\
             ML tools supporting reproducibility:\n\
             - ml_kmeans: Uses seed for centroid initialization\n\
             - ml_random_forest: Uses seed for bootstrap sampling and feature selection\n\
             - ml_tsne: Uses seed for initial embedding\n\n\
             Per-tool seeds override the global seed.",
            "=".repeat(40),
            seed_status
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Server Stats Tools
    // ========================================================================

    /// Get server memory and performance statistics.
    #[tool(
        description = "Get memory usage statistics for the analytics server. Shows dataset memory consumption, process memory, and memory trends. Useful for monitoring resource usage with large datasets."
    )]
    pub async fn server_stats(
        &self,
        Parameters(request): Parameters<ServerStatsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::{format_bytes, get_process_memory};

        let include_datasets = request.include_datasets.unwrap_or(true);
        let include_history = request.include_history.unwrap_or(false);

        let datasets = self.datasets.read().await;
        let profiler = self.memory_profiler.read().await;

        // Get detailed stats with dataset dimensions
        let stats = profiler.stats_with_datasets(datasets.iter().map(|(k, v)| (k.as_str(), v)));

        let mut output = String::new();
        output.push_str("Server Memory Statistics\n");
        output.push_str(&"=".repeat(50));
        output.push_str("\n\n");

        // Summary section
        output.push_str("## Summary\n\n");
        output.push_str(&format!(
            "Total Dataset Memory: {}\n",
            stats.dataset_memory_formatted
        ));
        output.push_str(&format!("Datasets Loaded: {}\n", stats.dataset_count));
        output.push_str(&format!(
            "Peak Memory: {}\n",
            format_bytes(profiler.peak_memory())
        ));
        output.push_str(&format!(
            "Uptime: {:.1}s\n",
            profiler.elapsed().as_secs_f64()
        ));

        // Process memory (if available)
        if let Some(ref proc_mem) = stats.process_memory {
            output.push_str(&format!("\nProcess RSS: {}\n", proc_mem.rss_formatted));
            output.push_str(&format!(
                "Process Virtual: {}\n",
                proc_mem.virtual_formatted
            ));
        } else if let Some(proc_mem) = get_process_memory() {
            output.push_str(&format!("\nProcess RSS: {}\n", proc_mem.rss_formatted));
            output.push_str(&format!(
                "Process Virtual: {}\n",
                proc_mem.virtual_formatted
            ));
        }

        // Per-dataset breakdown
        if include_datasets && !stats.datasets.is_empty() {
            output.push_str("\n## Datasets (by memory usage)\n\n");
            output.push_str(&format!(
                "{:<30} {:>12} {:>10} {:>10}\n",
                "Name", "Memory", "Rows", "Cols"
            ));
            output.push_str(&"-".repeat(64));
            output.push('\n');

            for ds in &stats.datasets {
                output.push_str(&format!(
                    "{:<30} {:>12} {:>10} {:>10}\n",
                    if ds.name.len() > 28 {
                        format!("{}...", &ds.name[..25])
                    } else {
                        ds.name.clone()
                    },
                    ds.memory_formatted,
                    ds.rows,
                    ds.columns
                ));
            }
        }

        // Memory history
        if include_history {
            let history = profiler.history();
            if !history.is_empty() {
                output.push_str("\n## Recent Memory History\n\n");
                output.push_str(&format!(
                    "{:>12} {:>15} {:>10}\n",
                    "Time (ms)", "Memory", "Datasets"
                ));
                output.push_str(&"-".repeat(40));
                output.push('\n');

                // Show last 10 entries
                let start = if history.len() > 10 {
                    history.len() - 10
                } else {
                    0
                };
                for snapshot in &history[start..] {
                    output.push_str(&format!(
                        "{:>12} {:>15} {:>10}\n",
                        snapshot.time_offset_ms,
                        format_bytes(snapshot.total_bytes),
                        snapshot.dataset_count
                    ));
                }
            }
        }

        drop(datasets);
        drop(profiler);

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Random Data Generation Tools
    // ========================================================================

    /// Generate random data with specified distributions.
    #[tool(
        description = "Generate a random dataset with specified columns and distributions. Supports: uniform (min, max), normal (mean, std), binomial (n, p), poisson (lambda), exponential (rate), bernoulli (p), categorical (categories, optional weights), uniform_int (min, max), sequence (start), constant (value), constant_string (value). Example column: {\"name\": \"x\", \"distribution\": {\"type\": \"normal\", \"mean\": 0, \"std\": 1}}"
    )]
    pub async fn generate_random_data(
        &self,
        Parameters(request): Parameters<GenerateRandomDataRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Bound the allocation before doing any work. Cells rather than rows,
        // because memory scales with the product: 5M rows is cheap at one column
        // and ruinous at fifty. Without a bound one anonymous request could
        // exhaust a 1 GB deployment, and since sessions live in process memory
        // that would take every other user's data with it. The budgets come from
        // the resource profile (hosted: generated_cells, generated_bytes,
        // string_bytes, array_length); without one (local, desktop, stdio)
        // generation is unconstrained, as it is for the CLI. A product that
        // overflows counts as u64::MAX, so it can never wrap back under a cap.
        const TOOL: &str = "generate_random_data";
        let limits = crate::tools::limits::current();
        let cells = request
            .n_rows
            .checked_mul(request.columns.len())
            .map_or(u64::MAX, |c| c as u64);
        if let Err(v) = limits.check_budget(TOOL, "generated_cells", cells, limits.generated_cells)
        {
            return Ok(budget_refusal(v));
        }

        // Parse column specifications
        let mut columns: Vec<ColumnSpec> = Vec::with_capacity(request.columns.len());

        for col_input in &request.columns {
            let dist: Distribution = match serde_json::from_value(col_input.distribution.clone()) {
                Ok(d) => d,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid distribution specification for column '{}': {}\n\n\
                         Expected format: {{\"type\": \"distribution_name\", ...params}}\n\
                         Available types:\n\
                         - uniform: {{\"type\": \"uniform\", \"min\": 0.0, \"max\": 1.0}}\n\
                         - normal: {{\"type\": \"normal\", \"mean\": 0.0, \"std\": 1.0}}\n\
                         - binomial: {{\"type\": \"binomial\", \"n\": 10, \"p\": 0.5}}\n\
                         - poisson: {{\"type\": \"poisson\", \"lambda\": 5.0}}\n\
                         - exponential: {{\"type\": \"exponential\", \"rate\": 1.0}}\n\
                         - bernoulli: {{\"type\": \"bernoulli\", \"p\": 0.5}}\n\
                         - categorical: {{\"type\": \"categorical\", \"categories\": [\"A\", \"B\", \"C\"], \"weights\": [0.5, 0.3, 0.2]}}\n\
                         - uniform_int: {{\"type\": \"uniform_int\", \"min\": 1, \"max\": 10}}\n\
                         - sequence: {{\"type\": \"sequence\", \"start\": 1}}\n\
                         - constant: {{\"type\": \"constant\", \"value\": 42.0}}\n\
                         - constant_string: {{\"type\": \"constant_string\", \"value\": \"text\"}}",
                        col_input.name, e
                    ))]));
                }
            };

            columns.push(ColumnSpec::new(&col_input.name, dist));
        }

        // Per-string and per-array caps, then the materialised size in bytes
        // (rows x per-column width, where string columns count their text).
        let string_cap = limits.cap(Family::StringBytes);
        let array_cap = limits.cap(Family::ArrayLength);
        for col in &columns {
            if let Err(refusal) =
                check_generated_column(&col.name, &col.distribution, string_cap, array_cap)
            {
                return Ok(refusal);
            }
        }
        let row_bytes = columns.iter().fold(0u64, |acc, c| {
            acc.saturating_add(generated_row_width(&c.distribution))
        });
        let requested_bytes = (request.n_rows as u64).saturating_mul(row_bytes);
        if let Err(v) = limits.check_budget(
            TOOL,
            "generated_bytes",
            requested_bytes,
            limits.generated_bytes,
        ) {
            return Ok(budget_refusal(v));
        }

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        // Generate the data
        let dataset = match generate_random_data(request.n_rows, columns, seed) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to generate random data: {}",
                    e
                ))]));
            }
        };

        // Get dataset info
        let n_rows = dataset.df().height();
        let n_cols = dataset.df().width();
        let col_names: Vec<String> = dataset
            .df()
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        // Determine dataset name
        let name = request.name.unwrap_or_else(|| "generated".to_string());

        // Store the dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "Random Dataset Generated\n{}\n\
             Name: {}\n\
             Rows: {}\n\
             Columns: {}\n\
             Column names: {}\n\
             Seed: {}\n\n\
             The dataset '{}' is now available for analysis.",
            "=".repeat(40),
            name,
            n_rows,
            n_cols,
            col_names.join(", "),
            seed.map(|s| s.to_string())
                .unwrap_or_else(|| "random".to_string()),
            name
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Report Generation Tools
    // ========================================================================

    /// Generate an HTML report from structured analysis results.
    #[tool(
        description = "Generate a self-contained HTML report from analysis results. The report includes proper styling, tables, charts (as embedded images), and is suitable for sharing or printing. Returns the complete HTML document as a string."
    )]
    pub async fn generate_report(
        &self,
        Parameters(request): Parameters<GenerateReportRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Build the report structure
        let mut report = HtmlReport::new(&request.title);

        if let Some(ref subtitle) = request.subtitle {
            report = report.with_subtitle(subtitle);
        }

        if let Some(ref author) = request.author {
            report = report.with_author(author);
        }

        // Process each section
        for section_input in &request.sections {
            let mut section = ReportSection::new(&section_input.title);

            for content_input in &section_input.content {
                match content_input.content_type.as_str() {
                    "text" => {
                        if let Some(ref text) = content_input.text {
                            section.add_text(text);
                        }
                    }
                    "code" => {
                        if let Some(ref code) = content_input.text {
                            section.add_code(code, content_input.language.as_deref());
                        }
                    }
                    "table" => {
                        if let (Some(headers), Some(rows)) =
                            (&content_input.headers, &content_input.rows)
                        {
                            let mut table = ReportTable::new(headers.clone());
                            if let Some(ref caption) = content_input.caption {
                                table = table.with_caption(caption);
                            }
                            for row in rows {
                                table.add_row(row.clone());
                            }
                            section.add_table(table);
                        }
                    }
                    "chart" => {
                        if let Some(ref image) = content_input.image_base64 {
                            section.add_chart(
                                image,
                                content_input.chart_title.as_deref(),
                                content_input.chart_caption.as_deref(),
                            );
                        }
                    }
                    "stats" => {
                        if let Some(ref stats) = content_input.stats {
                            let items: Vec<(String, String)> = stats
                                .iter()
                                .filter_map(|pair| {
                                    if pair.len() >= 2 {
                                        Some((pair[0].clone(), pair[1].clone()))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            section.add_statistics(items);
                        }
                    }
                    _ => {
                        // Unknown content type, skip
                    }
                }
            }

            report.add_section(section);
        }

        // Generate the HTML
        let html = report.to_html();

        // Return the HTML - it's quite long so we provide summary info
        let summary = format!(
            "HTML Report Generated\n\
             =====================\n\
             Title: {}\n\
             Sections: {}\n\
             HTML Length: {} characters\n\n\
             The complete HTML report follows:\n\n{}",
            request.title,
            request.sections.len(),
            html.len(),
            html
        );

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }

    // ========================================================================
    // Session Export/Import Tools
    // ========================================================================

    /// Export the current analysis session to a JSON file.
    #[tool(
        description = "Export the current session including all loaded datasets and their metadata. Can save to file or return as string. Useful for saving your analysis state to resume later."
    )]
    pub async fn export_session(
        &self,
        Parameters(request): Parameters<ExportSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        use std::io::Write;

        let limits = crate::tools::limits::current();
        let datasets = self.datasets.read().await;
        let include_data = request.include_data.unwrap_or(true);
        let created_at = chrono::Utc::now().to_rfc3339();

        if let Some(file_path) = request.file_path {
            let path = match path_jail::validate_data_path_for_write(&file_path) {
                Ok(p) => p,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Refused to export session: {}",
                        e
                    ))]));
                }
            };

            // Streamed to a temporary file in the validated folder, capped at
            // the export_bytes budget, then renamed into place.
            let cap = limits.export_bytes;
            let mut written = 0u64;
            let saved = path_jail::write_atomically(&path, |tmp| {
                let file = std::fs::File::create(tmp)
                    .map_err(|_| "Failed to write the session file".to_string())?;
                let mut w = CappedWriter::new(std::io::BufWriter::new(file), cap);
                let result = write_session_json(&datasets, include_data, &created_at, &mut w)
                    .map_err(|e| e.to_string())
                    .and_then(|()| w.flush().map_err(|e| e.to_string()));
                if w.exceeded() {
                    return Err(format!(
                        "The session export is larger than this server's export limit of {} \
                         bytes; export fewer or smaller datasets (export_dataset writes one \
                         dataset).",
                        cap.unwrap_or_default()
                    ));
                }
                written = w.written();
                result.map_err(|e| format!("Failed to write the session file: {e}"))
            });
            if let Err(e) = saved {
                return Ok(CallToolResult::error(vec![Content::text(e)]));
            }

            Ok(CallToolResult::success(vec![Content::text(format!(
                "Session exported successfully to: {}\n\
                 Datasets saved: {}\n\
                 Include data: {}\n\
                 Bytes written: {}",
                path_jail::display_path(&path),
                datasets.len(),
                include_data,
                written
            ))]))
        } else {
            // Inline: bounded by the smaller of the export and response budgets.
            let cap = match (limits.export_bytes, limits.response_bytes) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            let mut w = CappedWriter::new(Vec::new(), cap);
            let result = write_session_json(&datasets, include_data, &created_at, &mut w);
            if w.exceeded() {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "The session is too large to return inline (more than {} bytes); pass \
                     file_path to save it to a file.",
                    cap.unwrap_or_default()
                ))]));
            }
            if let Err(e) = result {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to serialise the session: {e}"
                ))]));
            }
            let json_output = match String::from_utf8(w.into_inner()) {
                Ok(s) => s,
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(
                        "Failed to serialise the session: the output is not UTF-8",
                    )]));
                }
            };
            Ok(CallToolResult::success(vec![Content::text(format!(
                "Session Export\n{}\n\
                 Datasets: {}\n\n{}",
                "=".repeat(40),
                datasets.len(),
                json_output
            ))]))
        }
    }

    /// Import a previously exported analysis session.
    #[tool(
        description = "Import a previously exported session from a JSON file. Can merge with existing session or replace it. Restores all datasets with their original names."
    )]
    pub async fn import_session(
        &self,
        Parameters(request): Parameters<ImportSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        let path = match path_jail::validate_data_path(&request.file_path) {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Refused to import session: {}",
                    e
                ))]));
            }
        };
        let shown = path_jail::display_path(&path);
        let read_failed = || {
            Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to read session file `{shown}`"
            ))]))
        };

        // Size-checked before the file is opened, then parsed in one
        // streaming pass under the same budget; nothing in the session
        // changes until the whole file has been read.
        let budget = crate::tools::limits::current().export_bytes;
        if let Some(cap) = budget {
            match std::fs::metadata(&path) {
                Ok(meta) if meta.len() > cap => {
                    tracing::info!(
                        tool = "import_session",
                        override_key = "export_bytes",
                        requested = meta.len(),
                        limit = cap,
                        "request refused by a named budget"
                    );
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "`{shown}` is larger than this server's import limit of {cap} bytes"
                    ))]));
                }
                Ok(_) => {}
                Err(_) => return read_failed(),
            }
        }
        let Ok(file) = std::fs::File::open(&path) else {
            return read_failed();
        };
        let parsed = match parse_session_json(std::io::BufReader::new(file), budget) {
            Ok(parsed) => parsed,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let merge = request.merge.unwrap_or(false);
        let mut datasets = self.datasets.write().await;

        if !merge {
            datasets.clear();
        }

        let mut imported_count = 0;
        let mut errors = Vec::new();

        for (name, dataset) in parsed {
            match build_dataset(dataset) {
                Ok(df) => {
                    datasets.insert(name, Dataset::new(df));
                    imported_count += 1;
                }
                Err(e) => errors.push(format!("{}: {}", name, e)),
            }
        }

        let mut output = format!(
            "Session Import\n{}\n\
             File: {}\n\
             Mode: {}\n\
             Datasets imported: {}\n",
            "=".repeat(40),
            shown,
            if merge { "merge" } else { "replace" },
            imported_count
        );

        if !errors.is_empty() {
            output.push_str(&format!("\nErrors:\n{}", errors.join("\n")));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::limits::{self, Family, Limits};
    use std::io::Write;

    #[test]
    fn capped_writer_stops_at_the_cap() {
        let mut w = CappedWriter::new(Vec::new(), Some(10));
        assert!(w.write_all(b"0123456789").is_ok());
        assert!(!w.exceeded());
        assert!(w.write_all(b"x").is_err());
        assert!(w.exceeded());
        assert_eq!(w.written(), 10);
        assert_eq!(w.into_inner(), b"0123456789");
        let mut unbounded = CappedWriter::new(Vec::new(), None);
        assert!(unbounded.write_all(&[0u8; 4096]).is_ok());
        assert!(!unbounded.exceeded());
    }

    use p2a_core::polars::prelude::df;

    fn mixed_datasets() -> HashMap<String, Dataset> {
        let df = df! {
            "f" => [Some(1.5), Some(f64::NAN), None],
            "i" => [Some(1i64), None, Some(3)],
            "s" => [Some("a"), None, Some("c\"q")],
            "b" => [Some(true), Some(false), None],
            "g" => [Some(0.1f32), None, Some(2.0f32)]
        }
        .unwrap();
        let other = df! { "z" => [10i64, 20], "a" => [0.5, 0.25] }.unwrap();
        HashMap::from([
            ("mixed".to_string(), Dataset::new(df)),
            ("other".to_string(), Dataset::new(other)),
        ])
    }

    /// Verbatim copy of the pre-change Value-tree builder of `export_session`
    /// (`created_at` aside), kept as the reference the streamed writer must
    /// reproduce.
    fn legacy_session_value(
        datasets: &HashMap<String, Dataset>,
        include_data: bool,
    ) -> serde_json::Value {
        use p2a_core::polars::prelude::*;

        let mut session_data = serde_json::Map::new();
        session_data.insert("version".to_string(), serde_json::json!("1.0"));
        session_data.insert(
            "created_at".to_string(),
            serde_json::json!(chrono::Utc::now().to_rfc3339()),
        );

        let mut datasets_json = serde_json::Map::new();
        for (name, dataset) in datasets.iter() {
            let df = dataset.df();
            let mut ds_info = serde_json::Map::new();

            // Save schema
            let schema: Vec<serde_json::Value> = df
                .get_columns()
                .iter()
                .map(|col| {
                    serde_json::json!({
                        "name": col.name().to_string(),
                        "dtype": format!("{:?}", col.dtype())
                    })
                })
                .collect();
            ds_info.insert("schema".to_string(), serde_json::json!(schema));
            ds_info.insert("n_rows".to_string(), serde_json::json!(df.height()));
            ds_info.insert("n_cols".to_string(), serde_json::json!(df.width()));

            if include_data {
                // Serialize actual data
                let mut columns_data = serde_json::Map::new();
                for col in df.get_columns() {
                    let col_name = col.name().to_string();
                    let values: Vec<serde_json::Value> = (0..col.len())
                        .map(|i| match col.get(i) {
                            Ok(av) => match av {
                                AnyValue::Null => serde_json::Value::Null,
                                AnyValue::Boolean(b) => serde_json::json!(b),
                                AnyValue::Int8(v) => serde_json::json!(v),
                                AnyValue::Int16(v) => serde_json::json!(v),
                                AnyValue::Int32(v) => serde_json::json!(v),
                                AnyValue::Int64(v) => serde_json::json!(v),
                                AnyValue::UInt8(v) => serde_json::json!(v),
                                AnyValue::UInt16(v) => serde_json::json!(v),
                                AnyValue::UInt32(v) => serde_json::json!(v),
                                AnyValue::UInt64(v) => serde_json::json!(v),
                                AnyValue::Float32(v) => serde_json::json!(v),
                                AnyValue::Float64(v) => serde_json::json!(v),
                                AnyValue::String(s) => serde_json::json!(s),
                                _ => serde_json::json!(format!("{:?}", av)),
                            },
                            Err(_) => serde_json::Value::Null,
                        })
                        .collect();
                    columns_data.insert(col_name, serde_json::json!(values));
                }
                ds_info.insert("data".to_string(), serde_json::json!(columns_data));
            }

            datasets_json.insert(name.clone(), serde_json::json!(ds_info));
        }
        session_data.insert("datasets".to_string(), serde_json::json!(datasets_json));
        serde_json::Value::Object(session_data)
    }

    #[test]
    fn streamed_session_json_matches_legacy_tree() {
        let ds = mixed_datasets();
        for include_data in [true, false] {
            let mut out = Vec::new();
            write_session_json(&ds, include_data, "2026-01-01T00:00:00Z", &mut out).unwrap();
            let mut got: serde_json::Value = serde_json::from_slice(&out).unwrap();
            let mut want = legacy_session_value(&ds, include_data);
            assert_eq!(got["created_at"], "2026-01-01T00:00:00Z");
            got.as_object_mut().unwrap().remove("created_at");
            want.as_object_mut().unwrap().remove("created_at");
            assert_eq!(got, want, "include_data={include_data}");
        }
        // Datasets in name order, top-level keys in a fixed order.
        let mut out = Vec::new();
        write_session_json(&ds, false, "t", &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with(r#"{"version":"1.0","created_at":"t","datasets":{"mixed""#));
        assert!(text.find(r#""mixed""#) < text.find(r#""other""#));
    }

    /// Runs `export_session` under `limits` on a server holding `datasets`.
    fn export_session_under(
        limits: Limits,
        datasets: HashMap<String, Dataset>,
        request: ExportSessionRequest,
    ) -> CallToolResult {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = AnalyticsServer::new();
        rt.block_on(async { server.datasets.write().await.extend(datasets) });
        limits::with_limits(limits, || {
            rt.block_on(server.export_session(Parameters(request)))
        })
        .expect("refusals are in-band tool errors")
    }

    #[test]
    fn export_session_inline_refuses_large_sessions() {
        // response_bytes override 2,000; a 1,000-row numeric dataset serialises to well over that.
        let limits = Limits::off()
            .with_overrides(&["response_bytes=2000".into()])
            .unwrap();
        let big = generate_random_data(
            1_000,
            vec![ColumnSpec::new(
                "a",
                Distribution::Normal {
                    mean: 0.0,
                    std: 1.0,
                },
            )],
            Some(1),
        )
        .unwrap();
        let r = export_session_under(
            limits,
            HashMap::from([("big".to_string(), big)]),
            ExportSessionRequest {
                file_path: None,
                include_data: Some(true),
            },
        );
        assert_eq!(r.is_error, Some(true));
        assert!(
            text(&r).contains("too large to return inline"),
            "{}",
            text(&r)
        );
        assert!(text(&r).contains("2000"), "{}", text(&r));
    }

    #[test]
    fn export_session_inline_small_session_parses() {
        let df = df! { "x" => [1.5, 2.25, 3.0], "name" => ["a", "b", "c"] }.unwrap();
        let r = export_session_under(
            Limits::off(),
            HashMap::from([("small".to_string(), Dataset::new(df))]),
            ExportSessionRequest {
                file_path: None,
                include_data: Some(true),
            },
        );
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
        let body = text(&r);
        let (header, json) = body.split_once("\n\n").unwrap();
        assert!(header.starts_with("Session Export"), "{header}");
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(
            v["datasets"]["small"]["data"]["x"],
            serde_json::json!([1.5, 2.25, 3.0])
        );
        assert_eq!(
            v["datasets"]["small"]["data"]["name"],
            serde_json::json!(["a", "b", "c"])
        );
        assert_eq!(v["datasets"]["small"]["n_rows"], 3);
    }

    /// Verbatim copy of the pre-change column conversion of `import_session`,
    /// applied to a parsed `Value`, as the reference for the streaming parser.
    fn legacy_import(json: &str) -> Vec<(String, Result<DataFrame, String>)> {
        use p2a_core::polars::prelude::*;

        let session: serde_json::Value = serde_json::from_str(json).unwrap();
        let datasets_obj = session.get("datasets").and_then(|v| v.as_object()).unwrap();
        let mut out = Vec::new();
        for (name, ds_info) in datasets_obj {
            let ds_obj = match ds_info.as_object() {
                Some(obj) => obj,
                None => {
                    out.push((name.clone(), Err("invalid format".to_string())));
                    continue;
                }
            };

            // Check if we have data to restore
            if let Some(data) = ds_obj.get("data").and_then(|v| v.as_object()) {
                // Reconstruct DataFrame from stored columns
                let mut columns_vec: Vec<Column> = Vec::new();

                for (col_name, values) in data {
                    if let Some(arr) = values.as_array() {
                        // Try to determine column type from first non-null value
                        let first_non_null = arr.iter().find(|v| !v.is_null());

                        let series: Series = match first_non_null {
                            Some(serde_json::Value::Number(n)) if n.is_f64() => {
                                let vals: Vec<Option<f64>> =
                                    arr.iter().map(|v| v.as_f64()).collect();
                                Series::new(col_name.into(), vals)
                            }
                            Some(serde_json::Value::Number(_)) => {
                                let vals: Vec<Option<i64>> =
                                    arr.iter().map(|v| v.as_i64()).collect();
                                Series::new(col_name.into(), vals)
                            }
                            Some(serde_json::Value::Bool(_)) => {
                                let vals: Vec<Option<bool>> =
                                    arr.iter().map(|v| v.as_bool()).collect();
                                Series::new(col_name.into(), vals)
                            }
                            _ => {
                                // Default to string
                                let vals: Vec<Option<String>> = arr
                                    .iter()
                                    .map(|v| {
                                        if v.is_null() {
                                            None
                                        } else if let Some(s) = v.as_str() {
                                            Some(s.to_string())
                                        } else {
                                            Some(v.to_string())
                                        }
                                    })
                                    .collect();
                                Series::new(col_name.into(), vals)
                            }
                        };
                        columns_vec.push(series.into());
                    }
                }

                if !columns_vec.is_empty() {
                    match DataFrame::new(columns_vec) {
                        Ok(df) => out.push((name.clone(), Ok(df))),
                        Err(e) => out.push((name.clone(), Err(format!("DataFrame error - {}", e)))),
                    }
                } else {
                    out.push((name.clone(), Err("no column data found".to_string())));
                }
            } else {
                out.push((
                    name.clone(),
                    Err("no data field (metadata-only session)".to_string()),
                ));
            }
        }
        out
    }

    /// `df` with its columns in name order.
    fn sorted_columns(df: &DataFrame) -> DataFrame {
        let mut names: Vec<String> = df
            .get_column_names()
            .into_iter()
            .map(|n| n.to_string())
            .collect();
        names.sort();
        df.select(names).unwrap()
    }

    #[test]
    fn streaming_import_matches_legacy_conversion() {
        let json = r#"{"version":"1.0","datasets":{
          "a":{"data":{"lead_nulls":[null,null,1.5,2],"ints":[1,2.5,null],"mixed":["x",1,true,null],
                       "all_null":[null,null],"bools":[true,null,false],"not_array":5}},
          "b":{"schema":[]},
          "c":7,
          "d":{"data":{}},
          "e":{"schema":[{"name":"ignored"}],"data":{"lead_nulls":[null,null,1.5,2],
                       "ints":[1,2.5,null,4],"mixed":["x",1,true,null],"all_null":[null,null,null,null],
                       "bools":[true,null,false,true],"big":[1,18446744073709551615,-3,null],
                       "nested":[[1,2],{"k":"v"},null,"s"],"not_array":{"x":1}},"n_rows":4},
          "f":{"data":5}}}"#;
        let got: Vec<(String, Result<DataFrame, String>)> =
            parse_session_json(json.as_bytes(), None)
                .unwrap()
                .into_iter()
                .map(|(name, parsed)| (name, build_dataset(parsed)))
                .collect();
        let want = legacy_import(json);
        let names = |v: &[(String, Result<DataFrame, String>)]| {
            let mut n: Vec<String> = v.iter().map(|(k, _)| k.clone()).collect();
            n.sort();
            n
        };
        assert_eq!(names(&got), names(&want));
        // Kind of each error: the text before " - " (DataFrame errors name a column).
        let kind = |e: &str| e.split(" - ").next().unwrap().to_string();
        for (name, legacy) in &want {
            let (_, streamed) = got.iter().find(|(n, _)| n == name).unwrap();
            match (streamed, legacy) {
                (Ok(a), Ok(b)) => assert!(
                    sorted_columns(a).equals_missing(&sorted_columns(b)),
                    "{name}:\n{a}\n{b}"
                ),
                (Err(a), Err(b)) => assert_eq!(kind(a), kind(b), "{name}"),
                (a, b) => panic!("{name}: streamed {a:?}, legacy {b:?}"),
            }
        }
        // Columns keep their file order.
        let (_, e) = got.iter().find(|(n, _)| n == "e").unwrap();
        let order: Vec<String> = e
            .as_ref()
            .unwrap()
            .get_column_names()
            .into_iter()
            .map(|n| n.to_string())
            .collect();
        assert_eq!(
            order,
            [
                "lead_nulls",
                "ints",
                "mixed",
                "all_null",
                "bools",
                "big",
                "nested"
            ]
        );
    }

    #[test]
    fn streaming_import_refuses_files_without_datasets() {
        for json in [
            r#"{"version":"1.0"}"#,
            r#"{"datasets":[1,2]}"#,
            "[1,2]",
            "7",
        ] {
            let err = parse_session_json(json.as_bytes(), None).unwrap_err();
            assert!(err.contains("missing 'datasets'"), "{json}: {err}");
        }
        let err = parse_session_json(r#"{"datasets":{} trailing"#.as_bytes(), None).unwrap_err();
        assert!(err.starts_with("Invalid session file"), "{err}");
    }

    #[test]
    fn streaming_import_enforces_its_byte_budget() {
        let values: Vec<String> = (0..10_000).map(|i| i.to_string()).collect();
        let json = format!(
            r#"{{"datasets":{{"a":{{"data":{{"x":[{}]}}}}}}}}"#,
            values.join(",")
        );
        let err = parse_session_json(json.as_bytes(), Some(1_000)).unwrap_err();
        assert!(err.contains("import limit"), "{err}");
        assert!(err.contains("1000"), "{err}");
        // A nested element is charged while it is read, not after.
        let nested = format!(
            r#"{{"datasets":{{"a":{{"data":{{"x":[[{}]]}}}}}}}}"#,
            values.join(",")
        );
        let err = parse_session_json(nested.as_bytes(), Some(1_000)).unwrap_err();
        assert!(err.contains("import limit"), "{err}");
        // The same file passes with room to spare.
        assert!(parse_session_json(json.as_bytes(), Some(10_000_000)).is_ok());
    }

    #[test]
    fn streaming_import_keeps_file_order_and_the_last_duplicate() {
        let json =
            r#"{"datasets":{"a":1,"b":{"data":{"x":[1],"y":[2],"x":[3]}},"a":{"data":{"z":[4]}}}}"#;
        let parsed = parse_session_json(json.as_bytes(), None).unwrap();
        let names: Vec<&str> = parsed.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        let columns = |d: &ParsedDataset| match d {
            ParsedDataset::Columns(c) => c
                .iter()
                .map(|(n, c)| (n.clone(), format!("{c:?}")))
                .collect::<Vec<_>>(),
            other => panic!("{other:?}"),
        };
        assert_eq!(
            columns(&parsed[0].1),
            [("z".into(), "I64([Some(4)])".into())]
        );
        assert_eq!(
            columns(&parsed[1].1),
            [
                ("x".to_string(), "I64([Some(3)])".to_string()),
                ("y".to_string(), "I64([Some(2)])".to_string())
            ]
        );
    }

    /// Many dataset keys whose values are scalars never reach the per-column
    /// checkpoint; the per-dataset one must still stop the parse.
    #[test]
    fn streaming_import_of_many_dataset_keys_can_be_cancelled() {
        use p2a_core::resource::{Budget, CancelToken, with_budget};
        let keys: Vec<String> = (0..50_000).map(|i| format!(r#""d{i}":{i}"#)).collect();
        let json = format!(r#"{{"datasets":{{{}}}}}"#, keys.join(","));
        let token = CancelToken::new();
        token.cancel();
        let err = with_budget(Budget::new(None, token), || {
            parse_session_json(json.as_bytes(), None)
        })
        .unwrap_err();
        assert!(err.contains("stopped"), "{err}");
        assert!(!err.starts_with("Invalid session file"), "{err}");
        // Without a cancellation the same file parses, in file order.
        let parsed = parse_session_json(json.as_bytes(), None).unwrap();
        assert_eq!(parsed.len(), 50_000);
        assert_eq!(parsed[0].0, "d0");
        assert_eq!(parsed[49_999].0, "d49999");
        assert!(matches!(parsed[7].1, ParsedDataset::Invalid));
    }

    /// A path guaranteed to sit outside the jail root, whatever it resolves to:
    /// the root is either the process CWD (this crate's directory under the repo)
    /// or a `P2A_DATA_ROOT` another test set to its own tempdir. A fresh tempdir
    /// is outside both, so this needs no coordination with the process-global
    /// `DATA_ROOT` cache.
    fn canary_outside_jail() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("escaped.json");
        (dir, path.to_str().unwrap().to_string())
    }

    #[tokio::test]
    async fn export_session_refuses_path_outside_data_root() {
        let (_dir, target) = canary_outside_jail();
        let server = AnalyticsServer::new();

        let result = server
            .export_session(Parameters(ExportSessionRequest {
                file_path: Some(target.clone()),
                include_data: Some(false),
            }))
            .await
            .expect("handler should refuse in-band, not fail the call");

        assert!(
            !std::path::Path::new(&target).exists(),
            "export_session wrote outside the data root: {target}"
        );
        assert_eq!(result.is_error, Some(true), "refusal must be a tool error");
    }

    fn normal_column(name: &str) -> ColumnSpecInput {
        ColumnSpecInput {
            name: name.to_string(),
            distribution: serde_json::json!({"type": "normal", "mean": 0.0, "std": 1.0}),
        }
    }

    /// Runs `generate_random_data` under `limits`. The limits override is
    /// thread-local, so the handler runs on a current-thread runtime built here.
    fn generate_under(limits: Limits, request: GenerateRandomDataRequest) -> CallToolResult {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = AnalyticsServer::new();
        limits::with_limits(limits, || {
            rt.block_on(server.generate_random_data(Parameters(request)))
        })
        .expect("refusals are in-band tool errors")
    }

    /// The first content block's text.
    fn text(r: &CallToolResult) -> String {
        r.content
            .first()
            .and_then(|c| c.raw.as_text())
            .map(|t| t.text.clone())
            .unwrap_or_default()
    }

    fn constant_string(name: &str, len: usize) -> ColumnSpecInput {
        ColumnSpecInput {
            name: name.into(),
            distribution: serde_json::json!({"type": "constant_string", "value": "s".repeat(len)}),
        }
    }

    fn request(n_rows: usize, columns: Vec<ColumnSpecInput>) -> GenerateRandomDataRequest {
        GenerateRandomDataRequest {
            n_rows,
            columns,
            seed: Some(1),
            name: None,
        }
    }

    /// `n_rows` is an unbounded `usize` on an endpoint anonymous callers can
    /// reach, and p2a-core only rejects zero, so the hosted ceiling lives here.
    /// Sized just over the cap rather than absurdly high on purpose: an
    /// uncapped build has to be able to finish this allocation and return, or
    /// the test would hang instead of failing.
    #[test]
    fn generate_random_data_refuses_oversized_request() {
        let cells = Limits::hosted().generated_cells.unwrap();
        let r = generate_under(
            Limits::hosted(),
            request(
                cells as usize / 2 + 1,
                vec![normal_column("a"), normal_column("b")],
            ),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains(&cells.to_string()), "{}", text(&r));
    }

    #[test]
    fn generate_refuses_string_longer_than_string_bytes() {
        let cap = Limits::hosted().cap(Family::StringBytes).unwrap() as usize;
        let r = generate_under(
            Limits::hosted(),
            request(10, vec![constant_string("s", cap + 1)]),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains(&cap.to_string()), "{}", text(&r));
    }

    #[test]
    fn generate_budgets_bytes_not_cells() {
        // 60,000 rows x (16 + 4,000) bytes = 240,960,000 > 200 MiB, but only 60,000 cells.
        let limit = Limits::hosted().generated_bytes.unwrap();
        let r = generate_under(
            Limits::hosted(),
            request(60_000, vec![constant_string("s", 4_000)]),
        );
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains(&limit.to_string()), "{}", text(&r));
    }

    #[test]
    fn generate_refuses_too_many_categories() {
        let cap = Limits::hosted().cap(Family::ArrayLength).unwrap() as usize;
        let cats: Vec<String> = (0..=cap).map(|i| format!("c{i}")).collect();
        let col = ColumnSpecInput {
            name: "c".into(),
            distribution: serde_json::json!({"type": "categorical", "categories": cats}),
        };
        let r = generate_under(Limits::hosted(), request(10, vec![col]));
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains(&cap.to_string()), "{}", text(&r));
    }

    #[test]
    fn generate_is_unconstrained_without_a_profile() {
        let r = generate_under(
            Limits::off(),
            request(10, vec![constant_string("s", 10_000)]),
        );
        assert_ne!(r.is_error, Some(true), "{}", text(&r));
    }

    /// Guards the caps from being tightened into uselessness: an ordinary
    /// 10k x 2 simulation succeeds with and without the hosted profile.
    #[test]
    fn generate_random_data_allows_ordinary_request() {
        for limits in [Limits::off(), Limits::hosted()] {
            let r = generate_under(
                limits,
                request(10_000, vec![normal_column("a"), normal_column("b")]),
            );
            assert_ne!(
                r.is_error,
                Some(true),
                "an ordinary 10k x 2 request must still succeed: {}",
                text(&r)
            );
        }
    }

    #[tokio::test]
    async fn import_session_refuses_path_outside_data_root() {
        let (dir, target) = canary_outside_jail();
        // Valid session JSON, so a refusal can only come from the jail check —
        // not from a read failure or a parse error.
        std::fs::write(&target, r#"{"datasets":{}}"#).unwrap();
        let server = AnalyticsServer::new();

        let result = server
            .import_session(Parameters(ImportSessionRequest {
                file_path: target.clone(),
                merge: Some(false),
            }))
            .await
            .expect("handler should refuse in-band, not fail the call");

        assert_eq!(
            result.is_error,
            Some(true),
            "import_session read outside the data root: {}",
            dir.path().display()
        );
    }
}
