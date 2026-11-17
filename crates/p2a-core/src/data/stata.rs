//! Stata DTA file reader for formats 117 (Stata 13), 118 (Stata 14-18) and 119 (Stata 15-18, more
//! than 32,767 variables), little-endian (LSF) and big-endian (MSF).
//!
//! This is a pure Rust implementation of Stata's `dta` specification. The header is read tag by
//! tag; the `<map>` then locates every section (slot 2 `<variable_types>`, 3 `<varnames>`, 5
//! `<formats>`, 6 `<value_label_names>`, 9 `<data>`, 10 `<strls>`, 11 `<value_labels>`), and the
//! reader checks the tag at each offset instead of scanning for it. Field widths by release:
//!
//! | Release | `<K>` | `<N>` | label length | variable name, label set name | format | strL reference (v, o) |
//! |---|---|---|---|---|---|---|
//! | 117 | u16 | u32 | u8 | 33 bytes | 49 bytes | u32, u32 |
//! | 118 | u16 | u64 | u16 | 129 bytes | 57 bytes | u16, 6 bytes |
//! | 119 | u32 | u64 | u16 | 129 bytes | 57 bytes | u16, 6 bytes (haven); u24, 5 bytes (Stata) |
//!
//! **strL.** A strL cell holds a reference (v, o), the variable and observation (both from 1) of a
//! `GSO` block in `<strls>`: `"GSO"`, v u32, o (u32 in 117, u64 after), type u8 (130 text with a
//! trailing NUL, 129 binary), length u32, bytes. Equal values share one block. (0, 0) is "". A
//! reference the table does not hold is an error naming the column and row. Release 119
//! references are read with haven's 2/6 split, and with Stata's 3/5 split when that one does not
//! resolve. Binary blocks are read as text with invalid UTF-8 replaced, and a note says so.
//!
//! **Formats and value labels.** `%td` (and the legacy `%d`) columns load as `Date` (days since
//! 1960-01-01 less 3,653) and `%tc`/`%tC` as `Datetime(ms)` (milliseconds since 1960 less
//! 315,619,200,000; `%tC` leap seconds are kept, as haven does). Other `%t` formats (`%tw`, `%tm`,
//! `%tq`, `%th`, `%ty`, `%tb`) stay numeric, as does a `%td`/`%tc` column holding a non-integral
//! or out-of-range value; a note names each. Value-labelled columns stay numeric codes, and
//! [`load_stata_with`] returns their label sets; [`ValueLabels::Text`] turns them into text
//! (`haven::as_factor`: a labelled value becomes its label, another value its number).
//!
//! Text is UTF-8 in releases 118 and 119. Release 117 text is read as UTF-8 when it is valid UTF-8
//! (haven and ReadStat write UTF-8) and as Windows-1252 otherwise (Stata 13 writes the platform
//! encoding). A `str#` value ends at its first NUL and loses its trailing blanks, as haven reads
//! it (Stata itself keeps them). Values are pinned to `haven::read_dta` on the fixtures in
//! `validation/datasets/files/` (`validation/data/readers.md`). Formats before 117 (Stata 12 and
//! earlier) are refused with [`StataError::UnsupportedVersion`].
//!
//! Numeric storage types: byte, int and long load as Int64, float and double as Float64. Every
//! missing code (`.`, `.a`-`.z`) loads as null: byte above 100, int above 32,740, long above
//! 2,147,483,620, float above 0x7EFFFFFF (1.70141173319e38) and double above 0x7FDFFFFFFFFFFFFF
//! (8.98846567431e307), the format's largest values. A file must declare at least one variable.

use polars::frame::column::Column;
use polars::prelude::*;
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

use super::sas::{decode_utf8_or_windows_1252, decode_windows_1252_lenient};

/// Errors that can occur during Stata file reading.
#[derive(Error, Debug)]
pub enum StataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid Stata file: {0}")]
    InvalidFormat(String),

    #[error("Unsupported Stata version: {0}")]
    UnsupportedVersion(u8),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
}

/// Byte order for the file
#[derive(Debug, Clone, Copy, PartialEq)]
enum ByteOrder {
    LittleEndian, // LSF
    BigEndian,    // MSF
}

/// The three XML-tagged dta releases and their field widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Release {
    R117,
    R118,
    R119,
}

impl Release {
    fn k_width(self) -> usize {
        if self == Release::R119 { 4 } else { 2 }
    }
    fn n_width(self) -> usize {
        if self == Release::R117 { 4 } else { 8 }
    }
    fn label_length_width(self) -> usize {
        if self == Release::R117 { 1 } else { 2 }
    }
    fn name_len(self) -> usize {
        if self == Release::R117 { 33 } else { 129 }
    }
    fn format_len(self) -> usize {
        if self == Release::R117 { 49 } else { 57 }
    }
    /// Bytes of a strL reference's variable number (the rest is the observation).
    fn strl_v_width(self) -> usize {
        if self == Release::R117 { 4 } else { 2 }
    }
    /// Bytes of a GSO block's observation number.
    fn gso_o_width(self) -> usize {
        if self == Release::R117 { 4 } else { 8 }
    }
    /// Release 117 text may be Windows-1252; 118 and 119 are UTF-8.
    fn decode(self, bytes: &[u8], what: impl FnOnce() -> String) -> Result<String, StataError> {
        match self {
            Release::R117 => Ok(decode_utf8_or_windows_1252(bytes)),
            Release::R118 | Release::R119 => String::from_utf8(bytes.to_vec())
                .map_err(|_| StataError::ParseError(format!("{} is not valid UTF-8", what()))),
        }
    }
}

/// `<map>` slots (Stata dta specification, section 5.2).
const MAP_SLOTS: usize = 14;
const MAP_VARIABLE_TYPES: usize = 2;
const MAP_VARNAMES: usize = 3;
const MAP_SORTLIST: usize = 4;
const MAP_FORMATS: usize = 5;
const MAP_VALUE_LABEL_NAMES: usize = 6;
const MAP_DATA: usize = 9;
const MAP_STRLS: usize = 10;
const MAP_VALUE_LABELS: usize = 11;
const MAP_END: usize = 12;
const DATA_TAG: &[u8] = b"<data>";
const DATA_END_TAG: &[u8] = b"</data>";
/// The format's largest non-missing float and double; every missing code is above them.
const FLOAT_MAX: f32 = f32::from_bits(0x7EFF_FFFF);
const DOUBLE_MAX: f64 = f64::from_bits(0x7FDF_FFFF_FFFF_FFFF);
/// Rows between two resource checkpoints.
const ROWS_PER_CHECKPOINT: u64 = 1 << 16;
/// GSO blocks between two resource checkpoints.
const GSOS_PER_CHECKPOINT: usize = 1 << 12;
/// 1970-01-01 less 1960-01-01, in days and in milliseconds.
const DAYS_1960_TO_1970: i64 = 3_653;
const MS_1960_TO_1970: i64 = 315_619_200_000;

/// What a value-labelled column loads as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueLabels {
    /// The numeric codes (the default); the labels are reported in [`StataLoad::value_labels`].
    #[default]
    Codes,
    /// Text, as `haven::as_factor` gives it: a labelled value becomes its label, an unlabelled
    /// value its number ("3"), a missing value stays missing.
    Text,
}

crate::option_enum!(
    ValueLabels,
    crate::options::Choices {
        option: "value_labels",
        values: &[
            (&["codes"], ValueLabels::Codes),
            (&["text"], ValueLabels::Text),
        ],
    }
);

/// Options of [`load_stata_with`].
#[derive(Debug, Clone, Default)]
pub struct StataOptions {
    /// What value-labelled columns load as (default: their codes).
    pub value_labels: ValueLabels,
}

/// The value labels of one column.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnLabels {
    /// The column.
    pub column: String,
    /// The name of its label set in the file.
    pub label_set: String,
    /// (value, label) pairs in the file's order.
    pub labels: Vec<(f64, String)>,
}

/// A loaded Stata file: the frame, the value labels of its labelled columns, and notes on what
/// was not converted.
#[derive(Debug, Clone)]
pub struct StataLoad {
    /// The data.
    pub df: DataFrame,
    /// One entry per value-labelled column, in column order.
    pub value_labels: Vec<ColumnLabels>,
    /// What the reader left as it was, one sentence each (a `%tm` column kept numeric, say).
    pub notes: Vec<String>,
}

/// How a display format maps to a column type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatKind {
    Date,
    Datetime,
    /// Another `%t` format: kept numeric with a note.
    OtherTime,
    Plain,
}

impl FormatKind {
    fn of(format: &str) -> Self {
        let Some(rest) = format.strip_prefix('%') else {
            return FormatKind::Plain;
        };
        let rest = rest.strip_prefix('-').unwrap_or(rest);
        if rest.starts_with("td") || rest.starts_with('d') {
            FormatKind::Date
        } else if rest.starts_with("tc") || rest.starts_with("tC") {
            FormatKind::Datetime
        } else if rest.starts_with('t') {
            FormatKind::OtherTime
        } else {
            FormatKind::Plain
        }
    }
}

/// Stata variable type
#[derive(Debug, Clone)]
enum StataType {
    Byte,       // 1-byte signed integer (-127 to 100)
    Int,        // 2-byte signed integer
    Long,       // 4-byte signed integer
    Float,      // 4-byte float
    Double,     // 8-byte float
    Str(usize), // Fixed-length string
    StrL,       // Long string (reference)
}
impl StataType {
    fn from_code(code: u16) -> Result<Self, StataError> {
        match code {
            65530 => Ok(StataType::Byte),
            65529 => Ok(StataType::Int),
            65528 => Ok(StataType::Long),
            65527 => Ok(StataType::Float),
            65526 => Ok(StataType::Double),
            32768 => Ok(StataType::StrL),
            1..=2045 => Ok(StataType::Str(code as usize)),
            _ => Err(StataError::InvalidFormat(format!(
                "Unknown variable type code: {}",
                code
            ))),
        }
    }

    fn size(&self) -> usize {
        match self {
            StataType::Byte => 1,
            StataType::Int => 2,
            StataType::Long => 4,
            StataType::Float => 4,
            StataType::Double => 8,
            StataType::Str(len) => *len,
            StataType::StrL => 8, // Reference (v, o)
        }
    }
}

/// Variable metadata
#[derive(Debug, Clone)]
struct Variable {
    name: String,
    dtype: StataType,
    /// Display format (`%td`, `%9.0g`, ...).
    format: String,
    /// Name of the value-label set, "" for none.
    label_set: String,
}

/// Stata DTA file reader
pub struct StataReader<R: Read + Seek> {
    reader: BufReader<R>,
    byte_order: ByteOrder,
    release: Release,
    n_vars: usize,
    n_obs: u64,
    variables: Vec<Variable>,
    /// First byte of the rows (just after `<data>`).
    data_offset: u64,
    /// Bytes from `data_offset` to `<strls>`: the rows and `</data>`.
    data_len: u64,
    map: [u64; MAP_SLOTS],
    /// strL contents by (v, o), read before the rows when a strL column exists.
    strls: PlHashMap<(u64, u64), String>,
    notes: Vec<String>,
}

impl<R: Read + Seek> StataReader<R> {
    /// Create a new Stata reader from a file: header, `<map>`, variable types and names.
    pub fn new(reader: R) -> Result<Self, StataError> {
        let mut reader = BufReader::new(reader);
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;

        // Formats 102-116 are binary: byte 0 is the format, byte 1 the byte order (1 or 2).
        let mut first = [0u8; 2];
        if reader.read(&mut first)? == 2
            && (102..=116).contains(&first[0])
            && matches!(first[1], 1 | 2)
        {
            return Err(StataError::UnsupportedVersion(first[0]));
        }
        reader.seek(SeekFrom::Start(0))?;

        let (release, byte_order) = Self::read_header(&mut reader)?;
        let n_vars_raw = Self::read_uint(&mut reader, release.k_width(), byte_order)?;
        Self::expect(&mut reader, b"</K><N>")?;
        let n_obs = Self::read_uint(&mut reader, release.n_width(), byte_order)?;
        Self::expect(&mut reader, b"</N><label>")?;
        let label_len = Self::read_uint(&mut reader, release.label_length_width(), byte_order)?;
        Self::skip(&mut reader, label_len)?;
        Self::expect(&mut reader, b"</label><timestamp>")?;
        let timestamp_len = Self::read_uint(&mut reader, 1, byte_order)?;
        Self::skip(&mut reader, timestamp_len)?;
        Self::expect(&mut reader, b"</timestamp></header><map>")?;
        let mut map = [0u64; MAP_SLOTS];
        for slot in &mut map {
            *slot = Self::read_uint(&mut reader, 8, byte_order)?;
        }
        Self::expect(&mut reader, b"</map>")?;

        // A dataset without variables cannot hold observations (and the row loop would spin):
        // Stata refuses to save one, and ReadStat refuses to read one (DATA-73).
        if n_vars_raw == 0 {
            return Err(StataError::InvalidFormat(format!(
                "the file declares no variables (<K> = 0) and {n_obs} observations; a Stata \
                 dataset has at least one variable"
            )));
        }
        Self::validate_map(&map, file_len)?;
        let types_section = map[MAP_VARNAMES] - map[MAP_VARIABLE_TYPES];
        let names_section = map[MAP_SORTLIST] - map[MAP_VARNAMES];
        let types_fit = n_vars_raw
            .checked_mul(2)
            .and_then(|b| b.checked_add(b"<variable_types>".len() as u64))
            .is_some_and(|end| end <= types_section);
        let names_fit = n_vars_raw
            .checked_mul(release.name_len() as u64)
            .and_then(|b| b.checked_add(b"<varnames>".len() as u64))
            .is_some_and(|end| end <= names_section);
        if !types_fit || !names_fit {
            return Err(StataError::InvalidFormat(format!(
                "<K> = {n_vars_raw} variables do not fit in the <variable_types> and <varnames> \
                 sections; the file is truncated or corrupt"
            )));
        }
        let fits = |slot: usize, next: usize, width: usize, tag: &[u8]| {
            n_vars_raw
                .checked_mul(width as u64)
                .and_then(|b| b.checked_add(tag.len() as u64))
                .is_some_and(|end| end <= map[next].saturating_sub(map[slot]))
        };
        if !fits(
            MAP_FORMATS,
            MAP_VALUE_LABEL_NAMES,
            release.format_len(),
            b"<formats>",
        ) || !fits(
            MAP_VALUE_LABEL_NAMES,
            MAP_VALUE_LABEL_NAMES + 1,
            release.name_len(),
            b"<value_label_names>",
        ) {
            return Err(StataError::InvalidFormat(format!(
                "<K> = {n_vars_raw} variables do not fit in the <formats> and \
                 <value_label_names> sections; the file is truncated or corrupt"
            )));
        }
        let n_vars = usize::try_from(n_vars_raw)
            .map_err(|_| StataError::InvalidFormat(format!("<K> = {n_vars_raw} is too large")))?;

        Self::seek_tag(&mut reader, map[MAP_VARIABLE_TYPES], b"<variable_types>")?;
        let mut var_types = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            let code = Self::read_uint(&mut reader, 2, byte_order)? as u16;
            var_types.push(StataType::from_code(code)?);
        }

        Self::seek_tag(&mut reader, map[MAP_VARNAMES], b"<varnames>")?;
        let mut names = Vec::with_capacity(n_vars);
        for i in 0..n_vars {
            names.push(Self::read_fixed_string(
                &mut reader,
                release.name_len(),
                release,
                || format!("the name of variable {}", i + 1),
            )?);
        }
        Self::seek_tag(&mut reader, map[MAP_FORMATS], b"<formats>")?;
        let mut formats = Vec::with_capacity(n_vars);
        for name in &names {
            formats.push(Self::read_fixed_string(
                &mut reader,
                release.format_len(),
                release,
                || format!("the format of variable {name}"),
            )?);
        }
        Self::seek_tag(
            &mut reader,
            map[MAP_VALUE_LABEL_NAMES],
            b"<value_label_names>",
        )?;
        let mut variables = Vec::with_capacity(n_vars);
        for ((name, dtype), format) in names.into_iter().zip(var_types).zip(formats) {
            let label_set =
                Self::read_fixed_string(&mut reader, release.name_len(), release, || {
                    format!("the value-label name of variable {name}")
                })?;
            variables.push(Variable {
                name,
                dtype,
                format,
                label_set,
            });
        }

        Self::seek_tag(&mut reader, map[MAP_DATA], DATA_TAG)?;
        let data_offset = map[MAP_DATA] + DATA_TAG.len() as u64;
        let data_len = map[MAP_STRLS].saturating_sub(data_offset);

        Ok(StataReader {
            reader,
            byte_order,
            release,
            n_vars,
            n_obs,
            variables,
            data_offset,
            data_len,
            map,
            strls: PlHashMap::new(),
            notes: Vec::new(),
        })
    }

    /// Read the entire dataset into a Polars DataFrame, with value-labelled columns as their
    /// codes (see [`StataReader::read_with`] for the labels and notes).
    pub fn read_to_dataframe(self) -> Result<DataFrame, StataError> {
        Ok(self.read_with(&StataOptions::default())?.df)
    }

    /// Read the entire dataset, its value labels and the reader's notes.
    pub fn read_with(mut self, options: &StataOptions) -> Result<StataLoad, StataError> {
        let var_types: Vec<StataType> = self.variables.iter().map(|v| v.dtype.clone()).collect();
        let var_names: Vec<String> = self.variables.iter().map(|v| v.name.clone()).collect();

        if var_types.iter().any(|t| matches!(t, StataType::StrL)) {
            self.strls = self.read_strls()?;
        }
        self.reader.seek(SeekFrom::Start(self.data_offset))?;

        // Guard against a corrupt/malicious header declaring an implausible
        // observation count. `n_obs` comes straight from the file's `<N>` tag;
        // using it to pre-size the column buffers below would let a tiny file
        // trigger a multi-gigabyte allocation. The `<data>` section, located by
        // `<map>`, holds exactly the rows and `</data>`.
        let row_size: u64 = var_types.iter().map(|t| t.size() as u64).sum();
        let rows_end = DATA_END_TAG.len() as u64;
        let max_obs = self
            .data_len
            .saturating_sub(rows_end)
            .checked_div(row_size)
            .unwrap_or(0);
        let fits = self
            .n_obs
            .checked_mul(row_size)
            .and_then(|b| b.checked_add(rows_end))
            .is_some_and(|need| need <= self.data_len);
        if !fits {
            return Err(StataError::InvalidFormat(format!(
                "declared observation count {} exceeds what the file can hold \
                 ({max_obs} rows of {row_size} bytes in a {}-byte <data> section); file is \
                 truncated or corrupt",
                self.n_obs, self.data_len
            )));
        }

        // Column buffers, one per variable and typed by it, grown past their initial capacity as
        // rows are read. The guard above bounds n_obs by the file size; every cell the file
        // declares is also checked against the per-allocation ceiling.
        let n_obs = usize::try_from(self.n_obs).unwrap_or(usize::MAX);
        let capacity = n_obs.min(1 << 20);
        let cells = crate::resource::checked_len(&[self.n_vars, n_obs], "Stata columns")
            .map_err(resource_err)?;
        crate::resource::check_alloc::<Option<String>>(cells, "Stata columns")
            .map_err(resource_err)?;
        let mut columns = Vec::with_capacity(self.n_vars);
        for dtype in &var_types {
            columns.push(ColumnBuf::for_type(dtype, capacity)?);
        }

        for row in 0..self.n_obs {
            if row % ROWS_PER_CHECKPOINT == 0 {
                crate::resource::checkpoint().map_err(resource_err)?;
            }
            for (var_idx, dtype) in var_types.iter().enumerate() {
                self.read_value(dtype, &var_names[var_idx], row, &mut columns[var_idx])?;
            }
        }
        self.strls = PlHashMap::new();

        let label_sets = if self.variables.iter().any(|v| !v.label_set.is_empty()) {
            self.read_value_labels()?
        } else {
            HashMap::new()
        };
        let mut notes = std::mem::take(&mut self.notes);
        let mut value_labels = Vec::new();
        let mut polars_columns: Vec<Column> = Vec::with_capacity(columns.len());
        for (buf, var) in columns.into_iter().zip(&self.variables) {
            let numeric = !matches!(buf, ColumnBuf::Text(_));
            let mut column = buf.into_column(&var.name);
            let labels = if numeric && !var.label_set.is_empty() {
                match label_sets.get(&var.label_set) {
                    Some(labels) => Some(labels),
                    None => {
                        notes.push(format!(
                            "column '{}' names the value-label set '{}', which the file does not \
                             hold; it loads as numbers",
                            var.name, var.label_set
                        ));
                        None
                    }
                }
            } else {
                None
            };
            if let Some(labels) = labels {
                value_labels.push(ColumnLabels {
                    column: var.name.clone(),
                    label_set: var.label_set.clone(),
                    labels: labels.clone(),
                });
            }
            column = match (labels, options.value_labels) {
                (Some(labels), ValueLabels::Text) => labels_as_text(&column, labels)?,
                _ if numeric => apply_format(column, &var.format, &mut notes)?,
                _ => column,
            };
            polars_columns.push(column);
        }
        let df = DataFrame::new(polars_columns).map_err(StataError::PolarsError)?;
        Ok(StataLoad {
            df,
            value_labels,
            notes,
        })
    }

    /// The `<strls>` section: every GSO block's text by (v, o).
    fn read_strls(&mut self) -> Result<PlHashMap<(u64, u64), String>, StataError> {
        let start = self.map[MAP_STRLS];
        let end = self.map[MAP_VALUE_LABELS];
        if end < start {
            return Err(StataError::InvalidFormat(format!(
                "<map> places <value_labels> at byte {end}, before <strls> at byte {start}"
            )));
        }
        Self::seek_tag(&mut self.reader, start, b"<strls>")?;
        let order = self.byte_order;
        let release = self.release;
        let mut table = PlHashMap::new();
        let mut binary = false;
        // The position is tracked here: `stream_position` on the buffered reader is a system
        // call per block.
        let mut at = self.reader.stream_position()?;
        let header_len = 3 + 4 + release.gso_o_width() as u64 + 1 + 4;
        loop {
            let mut tag = [0u8; 3];
            self.reader.read_exact(&mut tag)?;
            if &tag == b"</s" {
                Self::expect(&mut self.reader, b"trls>")?;
                break;
            }
            if &tag != b"GSO" {
                return Err(StataError::InvalidFormat(format!(
                    "expected GSO or </strls> at byte {at}, found {:?}",
                    decode_windows_1252_lenient(&tag)
                )));
            }
            let v = Self::read_uint(&mut self.reader, 4, order)?;
            let o = Self::read_uint(&mut self.reader, release.gso_o_width(), order)?;
            let mut kind = [0u8; 1];
            self.reader.read_exact(&mut kind)?;
            let len = Self::read_uint(&mut self.reader, 4, order)?;
            let here = at + header_len;
            if here.checked_add(len).is_none_or(|stop| stop > end) {
                return Err(StataError::InvalidFormat(format!(
                    "the strL ({v}, {o}) at byte {at} declares {len} bytes, past the end of \
                     <strls>"
                )));
            }
            let len = len as usize; // at most the file's length
            let mut bytes: Vec<u8> = crate::resource::try_with_capacity(len, "Stata strL contents")
                .map_err(resource_err)?;
            bytes.resize(len, 0);
            self.reader.read_exact(&mut bytes)?;
            let text = match kind[0] {
                130 => {
                    if bytes.last() == Some(&0) {
                        bytes.pop();
                    }
                    match release {
                        // UTF-8 text is taken over without a copy.
                        Release::R118 | Release::R119 => {
                            String::from_utf8(bytes).map_err(|_| {
                                StataError::ParseError(format!(
                                    "the strL ({v}, {o}) is not valid UTF-8"
                                ))
                            })?
                        }
                        Release::R117 => {
                            release.decode(&bytes, || format!("the strL ({v}, {o})"))?
                        }
                    }
                }
                129 => {
                    binary = true;
                    String::from_utf8_lossy(&bytes).into_owned()
                }
                other => {
                    return Err(StataError::InvalidFormat(format!(
                        "the strL ({v}, {o}) at byte {at} has type {other} (129 or 130 expected)"
                    )));
                }
            };
            table.insert((v, o), text);
            at = here + len as u64;
            if table.len() % GSOS_PER_CHECKPOINT == 0 {
                crate::resource::checkpoint().map_err(resource_err)?;
            }
        }
        if binary {
            self.notes.push(
                "binary strL values were read as text, with invalid UTF-8 replaced".to_string(),
            );
        }
        Ok(table)
    }

    /// The `<value_labels>` section: each label set's (value, label) pairs by its name.
    fn read_value_labels(&mut self) -> Result<HashMap<String, Vec<(f64, String)>>, StataError> {
        let start = self.map[MAP_VALUE_LABELS];
        let end = self.map[MAP_END];
        if end < start {
            return Err(StataError::InvalidFormat(format!(
                "<map> places </stata_dta> at byte {end}, before <value_labels> at byte {start}"
            )));
        }
        Self::seek_tag(&mut self.reader, start, b"<value_labels>")?;
        let order = self.byte_order;
        let release = self.release;
        let mut sets = HashMap::new();
        loop {
            let at = self.reader.stream_position()?;
            let mut tag = [0u8; 5];
            self.reader.read_exact(&mut tag)?;
            if &tag == b"</val" {
                Self::expect(&mut self.reader, b"ue_labels>")?;
                break;
            }
            if &tag != b"<lbl>" {
                return Err(StataError::InvalidFormat(format!(
                    "expected <lbl> or </value_labels> at byte {at}, found {:?}",
                    decode_windows_1252_lenient(&tag)
                )));
            }
            let len = Self::read_uint(&mut self.reader, 4, order)?;
            let name =
                Self::read_fixed_string(&mut self.reader, release.name_len(), release, || {
                    "a value-label set name".to_string()
                })?;
            Self::skip(&mut self.reader, 3)?;
            let here = self.reader.stream_position()?;
            if here.checked_add(len).is_none_or(|stop| stop > end) || len < 8 {
                return Err(StataError::InvalidFormat(format!(
                    "the value-label set '{name}' at byte {at} declares {len} bytes, which do not \
                     fit <value_labels>"
                )));
            }
            let len = len as usize;
            let mut table: Vec<u8> = crate::resource::try_with_capacity(len, "Stata value labels")
                .map_err(resource_err)?;
            table.resize(len, 0);
            self.reader.read_exact(&mut table)?;
            Self::expect(&mut self.reader, b"</lbl>")?;
            let labels = parse_label_table(&table, order, release, &name)?;
            sets.insert(name, labels);
            crate::resource::checkpoint().map_err(resource_err)?;
        }
        Ok(sets)
    }

    /// `<stata_dta><header><release>NNN</release><byteorder>XXX</byteorder><K>`, read exactly.
    fn read_header(reader: &mut BufReader<R>) -> Result<(Release, ByteOrder), StataError> {
        Self::expect(reader, b"<stata_dta><header><release>")?;
        let mut release = [0u8; 3];
        reader.read_exact(&mut release)?;
        let release = match &release {
            b"117" => Release::R117,
            b"118" => Release::R118,
            b"119" => Release::R119,
            other => {
                let shown = decode_windows_1252_lenient(other);
                return Err(if !other.iter().all(u8::is_ascii_digit) {
                    StataError::InvalidFormat(format!("release {shown:?} is not a number"))
                } else {
                    match shown.parse::<u8>() {
                        Ok(v) => StataError::UnsupportedVersion(v),
                        Err(_) => StataError::InvalidFormat(format!(
                            "release {shown} is not supported (117, 118 and 119 are)"
                        )),
                    }
                });
            }
        };
        Self::expect(reader, b"</release><byteorder>")?;
        let mut order = [0u8; 3];
        reader.read_exact(&mut order)?;
        let byte_order = match &order {
            b"LSF" => ByteOrder::LittleEndian,
            b"MSF" => ByteOrder::BigEndian,
            _ => return Err(StataError::InvalidFormat("Invalid byte order".into())),
        };
        Self::expect(reader, b"</byteorder><K>")?;
        Ok((release, byte_order))
    }

    /// The next `tag.len()` bytes must be `tag`.
    fn expect(reader: &mut BufReader<R>, tag: &[u8]) -> Result<(), StataError> {
        let at = reader.stream_position()?;
        let shown = decode_windows_1252_lenient(tag);
        let mut buf = vec![0u8; tag.len()];
        reader.read_exact(&mut buf).map_err(|_| {
            StataError::InvalidFormat(format!(
                "expected {shown} at byte {at}, but the file ends first"
            ))
        })?;
        if buf != tag {
            return Err(StataError::InvalidFormat(format!(
                "expected {shown} at byte {at}, found {:?}",
                decode_windows_1252_lenient(&buf)
            )));
        }
        Ok(())
    }

    /// Seek to a `<map>` offset and check the section tag there.
    fn seek_tag(reader: &mut BufReader<R>, offset: u64, tag: &[u8]) -> Result<(), StataError> {
        reader.seek(SeekFrom::Start(offset))?;
        Self::expect(reader, tag)
    }

    fn skip(reader: &mut BufReader<R>, n: u64) -> Result<(), StataError> {
        let n = i64::try_from(n)
            .map_err(|_| StataError::InvalidFormat(format!("length {n} is too large")))?;
        reader.seek(SeekFrom::Current(n))?;
        Ok(())
    }

    /// The sections the reader uses lie inside the file, in the specification's order.
    fn validate_map(map: &[u64; MAP_SLOTS], file_len: u64) -> Result<(), StataError> {
        for (slot, tag) in [
            (MAP_VARIABLE_TYPES, "<variable_types>"),
            (MAP_VARNAMES, "<varnames>"),
            (MAP_SORTLIST, "<sortlist>"),
            (MAP_DATA, "<data>"),
            (MAP_STRLS, "<strls>"),
            (MAP_FORMATS, "<formats>"),
            (MAP_VALUE_LABEL_NAMES, "<value_label_names>"),
            (MAP_VALUE_LABEL_NAMES + 1, "<variable_labels>"),
        ] {
            if map[slot] > file_len {
                return Err(StataError::InvalidFormat(format!(
                    "<map> places {tag} at byte {}, past the end of the {file_len}-byte file",
                    map[slot]
                )));
            }
        }
        let ordered = map[MAP_VARIABLE_TYPES] < map[MAP_VARNAMES]
            && map[MAP_VARNAMES] < map[MAP_SORTLIST]
            && map[MAP_SORTLIST] <= map[MAP_DATA]
            && map[MAP_DATA] < map[MAP_STRLS];
        if !ordered {
            return Err(StataError::InvalidFormat(format!(
                "<map> sections are out of order (variable_types {}, varnames {}, sortlist {}, \
                 data {}, strls {})",
                map[MAP_VARIABLE_TYPES],
                map[MAP_VARNAMES],
                map[MAP_SORTLIST],
                map[MAP_DATA],
                map[MAP_STRLS]
            )));
        }
        let ordered = map[MAP_SORTLIST] <= map[MAP_FORMATS]
            && map[MAP_FORMATS] <= map[MAP_VALUE_LABEL_NAMES]
            && map[MAP_VALUE_LABEL_NAMES] <= map[MAP_VALUE_LABEL_NAMES + 1]
            && map[MAP_VALUE_LABEL_NAMES + 1] <= map[MAP_DATA];
        if !ordered {
            return Err(StataError::InvalidFormat(format!(
                "<map> sections are out of order (sortlist {}, formats {}, value_label_names {}, \
                 variable_labels {}, data {})",
                map[MAP_SORTLIST],
                map[MAP_FORMATS],
                map[MAP_VALUE_LABEL_NAMES],
                map[MAP_VALUE_LABEL_NAMES + 1],
                map[MAP_DATA]
            )));
        }
        Ok(())
    }

    /// An unsigned integer of `width` (1, 2, 4 or 8) bytes in the file's byte order.
    fn read_uint(
        reader: &mut BufReader<R>,
        width: usize,
        byte_order: ByteOrder,
    ) -> Result<u64, StataError> {
        let mut buf = [0u8; 8];
        let bytes = &mut buf[..width.min(8)];
        reader.read_exact(bytes)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => bytes
                .iter()
                .rev()
                .fold(0u64, |v, &b| (v << 8) | u64::from(b)),
            ByteOrder::BigEndian => bytes.iter().fold(0u64, |v, &b| (v << 8) | u64::from(b)),
        })
    }
    fn read_i8(reader: &mut BufReader<R>) -> Result<i8, StataError> {
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf)?;
        Ok(buf[0] as i8)
    }

    fn read_i16(reader: &mut BufReader<R>, byte_order: ByteOrder) -> Result<i16, StataError> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => i16::from_le_bytes(buf),
            ByteOrder::BigEndian => i16::from_be_bytes(buf),
        })
    }

    fn read_i32(reader: &mut BufReader<R>, byte_order: ByteOrder) -> Result<i32, StataError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => i32::from_le_bytes(buf),
            ByteOrder::BigEndian => i32::from_be_bytes(buf),
        })
    }

    fn read_f32(reader: &mut BufReader<R>, byte_order: ByteOrder) -> Result<f32, StataError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => f32::from_le_bytes(buf),
            ByteOrder::BigEndian => f32::from_be_bytes(buf),
        })
    }

    fn read_f64(reader: &mut BufReader<R>, byte_order: ByteOrder) -> Result<f64, StataError> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => f64::from_le_bytes(buf),
            ByteOrder::BigEndian => f64::from_be_bytes(buf),
        })
    }

    /// A NUL-terminated fixed-width string without its trailing blanks, as ReadStat reads it,
    /// decoded by the release's text rule.
    fn read_fixed_string(
        reader: &mut BufReader<R>,
        len: usize,
        release: Release,
        what: impl FnOnce() -> String,
    ) -> Result<String, StataError> {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;

        // Find null terminator
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let end = buf[..end]
            .iter()
            .rposition(|&b| b != b' ')
            .map_or(0, |i| i + 1);
        release.decode(&buf[..end], what)
    }
    /// Read one cell of `dtype` (row `row`, from 0) and push it to the variable's buffer.
    fn read_value(
        &mut self,
        dtype: &StataType,
        name: &str,
        row: u64,
        buf: &mut ColumnBuf,
    ) -> Result<(), StataError> {
        match (dtype, buf) {
            (StataType::Byte, ColumnBuf::Int(v)) => {
                let x = Self::read_i8(&mut self.reader)?;
                v.push((x <= 100).then_some(i64::from(x))); // 101..=127: . and .a-.z
            }
            (StataType::Int, ColumnBuf::Int(v)) => {
                let x = Self::read_i16(&mut self.reader, self.byte_order)?;
                v.push((x <= 32_740).then_some(i64::from(x)));
            }
            (StataType::Long, ColumnBuf::Int(v)) => {
                let x = Self::read_i32(&mut self.reader, self.byte_order)?;
                v.push((x <= 2_147_483_620).then_some(i64::from(x)));
            }
            (StataType::Float, ColumnBuf::Float(v)) => {
                let x = Self::read_f32(&mut self.reader, self.byte_order)?;
                v.push((!x.is_nan() && x <= FLOAT_MAX).then_some(f64::from(x)));
            }
            (StataType::Double, ColumnBuf::Float(v)) => {
                let x = Self::read_f64(&mut self.reader, self.byte_order)?;
                v.push((!x.is_nan() && x <= DOUBLE_MAX).then_some(x));
            }
            (StataType::Str(len), ColumnBuf::Text(v)) => {
                let release = self.release;
                let s = Self::read_fixed_string(&mut self.reader, *len, release, || {
                    format!("a value of variable {name}")
                })?;
                v.push(Some(s));
            }
            (StataType::StrL, ColumnBuf::Text(v)) => {
                let mut b = [0u8; 8];
                self.reader.read_exact(&mut b)?;
                v.push(Some(self.strl_text(&b, name, row)?));
            }
            _ => {
                return Err(StataError::ParseError(format!(
                    "internal: the buffer of variable {name} does not match its type"
                )));
            }
        }
        Ok(())
    }
}

impl<R: Read + Seek> StataReader<R> {
    /// The text a strL reference `b` (8 bytes in the file's byte order) points to.
    fn strl_text(&self, b: &[u8; 8], name: &str, row: u64) -> Result<String, StataError> {
        let split = |v_width: usize| -> (u64, u64) {
            let uint = |bytes: &[u8]| match self.byte_order {
                ByteOrder::LittleEndian => bytes
                    .iter()
                    .rev()
                    .fold(0u64, |acc, &x| (acc << 8) | u64::from(x)),
                ByteOrder::BigEndian => {
                    bytes.iter().fold(0u64, |acc, &x| (acc << 8) | u64::from(x))
                }
            };
            (uint(&b[..v_width]), uint(&b[v_width..]))
        };
        let (v, o) = split(self.release.strl_v_width());
        if (v, o) == (0, 0) {
            return Ok(String::new());
        }
        let resolves = |(v, o): (u64, u64)| {
            (1..=self.n_vars as u64).contains(&v) && (1..=self.n_obs).contains(&o)
        };
        if resolves((v, o))
            && let Some(text) = self.strls.get(&(v, o))
        {
            return Ok(text.clone());
        }
        // Stata's own release-119 references are 3/5; haven's are 2/6.
        if self.release == Release::R119 {
            let alt = split(3);
            if resolves(alt)
                && let Some(text) = self.strls.get(&alt)
            {
                return Ok(text.clone());
            }
        }
        Err(StataError::InvalidFormat(format!(
            "strL column '{name}', row {}: reference ({v}, {o}) is not in the file",
            row + 1
        )))
    }
}

/// The (value, label) pairs of a `<lbl>` table: n i32, text length i32, n text offsets i32, n
/// values i32, then the NUL-terminated labels.
fn parse_label_table(
    table: &[u8],
    order: ByteOrder,
    release: Release,
    name: &str,
) -> Result<Vec<(f64, String)>, StataError> {
    let i32_at = |at: usize| -> Option<i32> {
        let b: [u8; 4] = table.get(at..at + 4)?.try_into().ok()?;
        Some(match order {
            ByteOrder::LittleEndian => i32::from_le_bytes(b),
            ByteOrder::BigEndian => i32::from_be_bytes(b),
        })
    };
    let bad = || {
        StataError::InvalidFormat(format!(
            "the value-label set '{name}' is malformed (its counts or offsets do not fit its \
             table)"
        ))
    };
    let n = usize::try_from(i32_at(0).ok_or_else(bad)?).map_err(|_| bad())?;
    let txt_len = usize::try_from(i32_at(4).ok_or_else(bad)?).map_err(|_| bad())?;
    let txt_at = n
        .checked_mul(8)
        .and_then(|b| b.checked_add(8))
        .ok_or_else(bad)?;
    let text = table
        .get(txt_at..txt_at.checked_add(txt_len).ok_or_else(bad)?)
        .ok_or_else(bad)?;
    let mut labels =
        crate::resource::try_with_capacity(n, "Stata value labels").map_err(resource_err)?;
    for i in 0..n {
        let off = usize::try_from(i32_at(8 + 4 * i).ok_or_else(bad)?).map_err(|_| bad())?;
        let value = i32_at(8 + 4 * n + 4 * i).ok_or_else(bad)?;
        let rest = text.get(off..).ok_or_else(bad)?;
        let stop = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
        let label = release.decode(&rest[..stop], || format!("a label of set '{name}'"))?;
        labels.push((f64::from(value), label));
    }
    Ok(labels)
}

/// `column` with its display `format` applied: `%td` as Date, `%tc`/`%tC` as Datetime(ms);
/// other `%t` formats, and values a date cannot hold, keep the numbers with a note.
fn apply_format(
    column: Column,
    format: &str,
    notes: &mut Vec<String>,
) -> Result<Column, StataError> {
    let kind = FormatKind::of(format);
    let name = column.name().to_string();
    let (offset, target) = match kind {
        FormatKind::Plain => return Ok(column),
        FormatKind::OtherTime => {
            notes.push(format!(
                "column '{name}' has the Stata format {format}, which is not converted; it loads \
                 as numbers"
            ));
            return Ok(column);
        }
        FormatKind::Date => (DAYS_1960_TO_1970, DataType::Date),
        FormatKind::Datetime => (
            MS_1960_TO_1970,
            DataType::Datetime(TimeUnit::Milliseconds, None),
        ),
    };
    let values = column.cast(&DataType::Float64)?;
    let values = values.f64()?;
    let mut shifted: Vec<Option<i64>> =
        crate::resource::try_with_capacity(values.len(), "Stata dates").map_err(resource_err)?;
    // A Date holds i32 days; a millisecond Datetime holds i64.
    let limit = if kind == FormatKind::Date {
        f64::from(i32::MAX)
    } else {
        9.2e18
    };
    for v in values.into_iter() {
        match v {
            None => shifted.push(None),
            Some(x) if x.is_finite() && x.fract() == 0.0 && x.abs() <= limit => {
                shifted.push((x as i64).checked_sub(offset));
            }
            Some(_) => {
                notes.push(format!(
                    "column '{name}' has the Stata format {format} but holds a value that is not a \
                     whole number of {} in range; it loads as numbers",
                    if kind == FormatKind::Date {
                        "days"
                    } else {
                        "milliseconds"
                    }
                ));
                return Ok(column);
            }
        }
    }
    if kind == FormatKind::Date && shifted.iter().flatten().any(|&d| i32::try_from(d).is_err()) {
        notes.push(format!(
            "column '{name}' has the Stata format {format} but holds a day outside the range of a \
             date; it loads as numbers"
        ));
        return Ok(column);
    }
    Ok(Column::new(name.into(), shifted).cast(&target)?)
}

/// `column`'s values as text by `labels` (`haven::as_factor`): a labelled value becomes its
/// label, any other value its number, a missing value stays missing.
fn labels_as_text(column: &Column, labels: &[(f64, String)]) -> Result<Column, StataError> {
    let values = column.cast(&DataType::Float64)?;
    let values = values.f64()?;
    let mut text: Vec<Option<String>> =
        crate::resource::try_with_capacity(values.len(), "Stata value labels")
            .map_err(resource_err)?;
    for v in values.into_iter() {
        text.push(v.map(|x| {
            labels
                .iter()
                .find(|(value, _)| *value == x)
                .map(|(_, label)| label.clone())
                .unwrap_or_else(|| {
                    if x.fract() == 0.0 && x.abs() < 1e15 {
                        format!("{}", x as i64)
                    } else {
                        format!("{x}")
                    }
                })
        }));
    }
    Ok(Column::new(column.name().clone(), text))
}

/// The values of one variable, typed by its storage type.
enum ColumnBuf {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
    Text(Vec<Option<String>>),
}

impl ColumnBuf {
    fn for_type(dtype: &StataType, capacity: usize) -> Result<Self, StataError> {
        let what = "Stata columns";
        Ok(match dtype {
            StataType::Byte | StataType::Int | StataType::Long => ColumnBuf::Int(
                crate::resource::try_with_capacity(capacity, what).map_err(resource_err)?,
            ),
            StataType::Float | StataType::Double => ColumnBuf::Float(
                crate::resource::try_with_capacity(capacity, what).map_err(resource_err)?,
            ),
            StataType::Str(_) | StataType::StrL => ColumnBuf::Text(
                crate::resource::try_with_capacity(capacity, what).map_err(resource_err)?,
            ),
        })
    }

    fn into_column(self, name: &str) -> Column {
        match self {
            ColumnBuf::Int(v) => Column::new(name.into(), v),
            ColumnBuf::Float(v) => Column::new(name.into(), v),
            ColumnBuf::Text(v) => Column::new(name.into(), v),
        }
    }
}

/// Map a core resource error (allocation size, cancellation) to a parse error, as the SAS reader
/// does: a large valid file is not a corrupt one.
fn resource_err(e: crate::errors::EconError) -> StataError {
    StataError::ParseError(e.to_string())
}

/// Load a Stata DTA file into a Polars DataFrame, with value-labelled columns as their codes.
pub fn load_stata(path: impl AsRef<Path>) -> Result<DataFrame, StataError> {
    Ok(load_stata_with(path, &StataOptions::default())?.df)
}

/// Load a Stata DTA file with `options`, returning the frame, the value labels of its labelled
/// columns, and notes on what the reader did not convert.
pub fn load_stata_with(
    path: impl AsRef<Path>,
    options: &StataOptions,
) -> Result<StataLoad, StataError> {
    let file = std::fs::File::open(path)?;
    StataReader::new(file)?.read_with(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::sas::fixture_check::{assert_matches_haven, fixture_bytes, fixture_path, put};
    use p2a_validation::{Case, Check, Tol};
    use std::io::Cursor;

    fn load(b: Vec<u8>) -> Result<DataFrame, StataError> {
        StataReader::new(Cursor::new(b))?.read_to_dataframe()
    }

    fn assert_invalid(r: Result<DataFrame, StataError>, needle: &str) {
        match r {
            Err(StataError::InvalidFormat(m)) => assert!(m.contains(needle), "{m}"),
            other => panic!(
                "expected InvalidFormat mentioning {needle:?}, got {:?}",
                other.map(|d| d.shape())
            ),
        }
    }

    // dta118_haven.dta (haven::write_dta(version = 14), LSF): the <K> value (u16) at byte 70, the
    // <N> value (u64) at 79, the 14 <map> slots (u64) from byte 163.
    const K118: usize = 70;
    const N118: usize = 79;
    const MAP118: usize = 163;

    #[test]
    fn test_stata_type_from_code() {
        assert!(matches!(StataType::from_code(65530), Ok(StataType::Byte)));
        assert!(matches!(StataType::from_code(65526), Ok(StataType::Double)));
        assert!(matches!(StataType::from_code(100), Ok(StataType::Str(100))));
    }

    #[test]
    fn test_stata_invalid_magic() {
        assert!(StataReader::new(Cursor::new(b"Not a valid Stata file".to_vec())).is_err());
    }

    #[test]
    fn test_stata_fixture_offsets() {
        // The constants above are this layout; a regenerated fixture with another layout fails here.
        let b = fixture_bytes("dta118_haven.dta");
        assert_eq!(&b[K118 - 3..K118], b"<K>");
        assert_eq!(&b[N118 - 3..N118], b"<N>");
        assert_eq!(&b[MAP118 - 5..MAP118], b"<map>");
        assert_eq!(
            u64::from_le_bytes(b[MAP118 + 8..MAP118 + 16].try_into().unwrap()),
            158
        );
        let b = fixture_bytes("dta119_haven.dta");
        assert_eq!(&b[67..70], b"<K>");
        assert_eq!(&b[70..74], &[6, 0, 0, 0]); // release 119: <K> is a u32
    }

    // ---- R-pinned: haven::read_dta on the committed fixtures (Tol::exact) ----

    #[test]
    fn test_validate_read_haven_dta_118() {
        // DATA-22: <map> slot 2 is <variable_types>, 3 <varnames>, 9 <data>; <K> and <N> are in
        // the header before <map>.
        let case = Case::load("data_readers_values");
        let df = load_stata(fixture_path("dta118_haven.dta")).unwrap();
        assert_matches_haven(&case, "dta118_haven", &df);
    }

    #[test]
    fn test_validate_read_haven_dta_118_stata_written() {
        // DATA-22, DATA-50: written by Stata 14 itself (4 float columns, variable labels, str10).
        let case = Case::load("data_readers_values");
        let df = load_stata(fixture_path("dta118_stata_iris.dta")).unwrap();
        assert_matches_haven(&case, "dta118_stata_iris", &df);
    }

    #[test]
    fn test_validate_read_haven_dta_117() {
        // DATA-51: release 117 stores <N> in 4 bytes and the label length in 1; haven writes UTF-8.
        let case = Case::load("data_readers_values");
        let df = load_stata(fixture_path("dta117_haven.dta")).unwrap();
        assert_matches_haven(&case, "dta117_haven", &df);
    }

    #[test]
    fn test_validate_read_haven_dta_117_windows_1252() {
        // DATA-51: Stata 13 writes str# text in Windows-1252 ("Z\u{fc}rich" = 5A FC 72 69 63 68,
        // "\u{20ac}5" = 80 35); haven reads it with encoding = "windows-1252".
        let case = Case::load("data_readers_values");
        let df = load_stata(fixture_path("dta117_latin1.dta")).unwrap();
        assert_matches_haven(&case, "dta117_latin1", &df);
    }

    #[test]
    fn test_validate_read_haven_dta_119() {
        // DATA-51: release 119 stores <K> in 4 bytes.
        let case = Case::load("data_readers_values");
        let df = load_stata(fixture_path("dta119_haven.dta")).unwrap();
        assert_matches_haven(&case, "dta119_haven", &df);
    }

    #[test]
    fn test_validate_read_haven_dta_119_big_endian() {
        // DATA-51: an MSF 119 file; read as u16, its <K> would be 0.
        let case = Case::load("data_readers_values");
        let df = load_stata(fixture_path("dta119_msf.dta")).unwrap();
        assert_matches_haven(&case, "dta119_msf", &df);
    }

    #[test]
    fn test_stata_119_k_is_four_bytes() {
        // DATA-51: byte 72 is the third byte of release 119's u32 <K>: K = 6 + 65536 = 65542,
        // which does not fit the sections (a u16 read sees K = 6 and then misreads the header).
        let mut b = fixture_bytes("dta119_haven.dta");
        b[72] = 1;
        assert_invalid(load(b), "<K> = 65542 variables do not fit");
    }

    #[test]
    fn test_validate_read_haven_dta_118_storage_types() {
        // byte, int, long, float, double at their valid extremes (100, 32740, 2147483620,
        // 1.7014117331926443e38 = 0x7EFFFFFF, 8.9884656743115785e307 = 0x7FDFFFFFFFFFFFFF are
        // values; ., .a, .b, .c, .z, .y are null).
        let case = Case::load("data_readers_values");
        let df = load_stata(fixture_path("dta118_types.dta")).unwrap();
        assert_matches_haven(&case, "dta118_types", &df);
        let dtypes: Vec<DataType> = df.get_columns().iter().map(|c| c.dtype().clone()).collect();
        assert_eq!(
            dtypes,
            [
                DataType::Int64,
                DataType::Int64,
                DataType::Int64,
                DataType::Float64,
                DataType::Float64
            ]
        );
    }

    #[test]
    fn test_validate_dataloader_loads_dta_fixture() {
        // The path `p2a data load` and MCP load_dataset take.
        let case = Case::load("data_readers_values");
        let ds = crate::data::DataLoader::load(fixture_path("dta118_haven.dta")).unwrap();
        assert_eq!(ds.name(), Some("dta118_haven"));
        assert_matches_haven(&case, "dta118_haven", ds.df());
    }

    // ---- formats, value labels and strL (haven::read_dta with formats and labels kept) ----

    /// Offsets of `bytes` (a valid fixture): the first row's byte and each variable's byte in a row.
    fn row_layout(bytes: &[u8]) -> (usize, usize, Vec<usize>) {
        let reader = StataReader::new(Cursor::new(bytes.to_vec())).unwrap();
        let mut offsets = Vec::new();
        let mut at = 0;
        for v in &reader.variables {
            offsets.push(at);
            at += v.dtype.size();
        }
        (reader.data_offset as usize, at, offsets)
    }

    fn column_index(bytes: &[u8], name: &str) -> usize {
        let reader = StataReader::new(Cursor::new(bytes.to_vec())).unwrap();
        reader
            .variables
            .iter()
            .position(|v| v.name == name)
            .unwrap()
    }

    fn assert_text_matches(case: &Case, key: &str, col: &Column) {
        let want = case.names(key);
        let bytes = case.vector(key);
        let got: Vec<Option<&str>> = col.str().unwrap().into_iter().collect();
        assert_eq!(got.len(), want.len(), "{key}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(w.len() as f64, bytes[i], "{key}[{i}]: case JSON");
            assert_eq!(*g, Some(w.as_str()), "{key}[{i}]");
        }
    }

    fn check_strl(stem: &str) {
        let case = Case::load("data_readers_formats_values");
        let load = load_stata_with(
            fixture_path(&format!("{stem}.dta")),
            &StataOptions::default(),
        )
        .unwrap();
        assert_text_matches(&case, &format!("{stem}__sl"), load.df.column("sl").unwrap());
        assert!(
            load.notes.iter().all(|n| !n.contains("strL")),
            "{:?}",
            load.notes
        );
    }

    #[test]
    fn test_validate_read_dta_strl_117() {
        // DATA-74: before the fix every strL cell was the placeholder "<strL>".
        check_strl("dta117_formats");
    }

    #[test]
    fn test_validate_read_dta_strl_118() {
        check_strl("dta118_formats");
    }

    #[test]
    fn test_validate_read_dta_strl_119() {
        check_strl("dta119_formats");
    }

    #[test]
    fn test_validate_read_dta_strl_119_big_endian() {
        check_strl("dta119_formats_msf");
    }

    fn check_formats(stem: &str) {
        let case = Case::load("data_readers_formats_values");
        let path = fixture_path(&format!("{stem}.dta"));
        let load = load_stata_with(&path, &StataOptions::default()).unwrap();
        let df = &load.df;
        let mut check = Check::new(&case);
        let as_f64 = |name: &str, via: &DataType| -> Vec<f64> {
            df.column(name)
                .unwrap()
                .cast(via)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .map(|v| v.unwrap_or(f64::NAN))
                .collect()
        };
        assert_eq!(df.column("dt").unwrap().dtype(), &DataType::Date, "{stem}");
        assert_eq!(
            df.column("tc").unwrap().dtype(),
            &DataType::Datetime(TimeUnit::Milliseconds, None),
            "{stem}"
        );
        assert_eq!(
            df.column("mo").unwrap().dtype(),
            &DataType::Float64,
            "{stem}"
        );
        assert_eq!(df.column("gl").unwrap().dtype(), &DataType::Int64, "{stem}");
        assert_eq!(
            df.column("gd").unwrap().dtype(),
            &DataType::Float64,
            "{stem}"
        );
        check
            .vector(
                &format!("{stem}__dt"),
                &as_f64("dt", &DataType::Int32),
                Tol::exact(),
            )
            .vector(
                &format!("{stem}__tc"),
                &as_f64("tc", &DataType::Int64),
                Tol::exact(),
            )
            .vector(
                &format!("{stem}__mo"),
                &as_f64("mo", &DataType::Float64),
                Tol::exact(),
            )
            .vector(
                &format!("{stem}__id"),
                &as_f64("id", &DataType::Float64),
                Tol::exact(),
            )
            .vector(
                &format!("{stem}__gl"),
                &as_f64("gl", &DataType::Float64),
                Tol::exact(),
            )
            .vector(
                &format!("{stem}__gd"),
                &as_f64("gd", &DataType::Float64),
                Tol::exact(),
            );
        assert!(
            load.notes
                .iter()
                .any(|n| n.contains("'mo'") && n.contains("%tm")),
            "{stem}: {:?}",
            load.notes
        );
        // The label sets, in the file's order.
        assert_eq!(load.value_labels.len(), 2, "{stem}");
        for labels in &load.value_labels {
            let key = format!("{stem}__labels_{}", labels.column);
            assert_eq!(labels.label_set, labels.column, "{stem}");
            let names: Vec<&str> = labels.labels.iter().map(|(_, l)| l.as_str()).collect();
            assert_eq!(names, case.names(&key), "{key}");
            let values: Vec<f64> = labels.labels.iter().map(|(v, _)| *v).collect();
            check.vector(&key, &values, Tol::exact());
        }
        check.finish();
        // The text option: haven's as_factor.
        let text = load_stata_with(
            &path,
            &StataOptions {
                value_labels: ValueLabels::Text,
            },
        )
        .unwrap();
        for name in ["gl", "gd"] {
            let key = format!("{stem}__{name}_text");
            let col = text.df.column(name).unwrap();
            assert_eq!(col.dtype(), &DataType::String, "{key}");
            let missing = case.vector(&key);
            let got: Vec<Option<&str>> = col.str().unwrap().into_iter().collect();
            for (i, want) in case.names(&key).iter().enumerate() {
                if missing[i] == 1.0 {
                    assert_eq!(got[i], None, "{key}[{i}]");
                } else {
                    assert_eq!(got[i], Some(want.as_str()), "{key}[{i}]");
                }
            }
        }
        assert_eq!(text.value_labels, load.value_labels, "{stem}");
        // Every format haven reports is the file's.
        let reader = StataReader::new(std::fs::File::open(&path).unwrap()).unwrap();
        let formats: Vec<&str> = reader.variables.iter().map(|v| v.format.as_str()).collect();
        assert_eq!(formats, case.names(&format!("{stem}__formats")), "{stem}");
    }

    #[test]
    fn test_validate_read_dta_formats_117() {
        // DATA-75: before the fix dt loaded as Float64 day counts from 1960 and labels were lost.
        check_formats("dta117_formats");
    }

    #[test]
    fn test_validate_read_dta_formats_118() {
        check_formats("dta118_formats");
    }

    #[test]
    fn test_validate_read_dta_formats_119() {
        check_formats("dta119_formats");
    }

    #[test]
    fn test_validate_read_dta_formats_119_big_endian() {
        check_formats("dta119_formats_msf");
    }

    #[test]
    fn test_validate_data_readers_t4_manifest_lists_every_fixture() {
        let case = Case::load("data_readers_formats_values");
        let manifest = std::fs::read_to_string(case.dataset_path()).unwrap();
        for stem in [
            "dta117_formats",
            "dta118_formats",
            "dta119_formats",
            "dta119_formats_msf",
        ] {
            assert!(manifest.contains(&format!("\"{stem}.dta\"")), "{stem}");
            assert!(case.keys().any(|k| k == format!("{stem}__sl")), "{stem}");
        }
    }

    #[test]
    fn test_stata_strl_unresolved_reference_is_an_error() {
        // The first sl cell of dta118_formats.dta points to (7, 1); (7, 99) is not in <strls>.
        let b = fixture_bytes("dta118_formats.dta");
        let (data, row, offsets) = row_layout(&b);
        let at = data + offsets[column_index(&b, "sl")];
        assert_eq!(&b[at..at + 8], &[7, 0, 1, 0, 0, 0, 0, 0]);
        let mut patched = b.clone();
        put(&mut patched, at, &[7, 0, 99, 0, 0, 0, 0, 0]);
        assert_invalid(
            load(patched),
            "strL column 'sl', row 1: reference (7, 99) is not in the file",
        );
        // (0, 0) is the empty string.
        let mut empty = b.clone();
        put(&mut empty, at, &[0; 8]);
        let df = load(empty).unwrap();
        assert_eq!(df.column("sl").unwrap().str().unwrap().get(0), Some(""));
        assert_eq!(row, 4 + 8 + 8 + 4 + 8 + 8 + 8, "the fixture's row layout");
    }

    #[test]
    fn test_stata_strl_119_stata_split_resolves() {
        // Stata describes release 119 references as v in 3 bytes and o in 5 (pandas reads them
        // so); haven writes and reads 2/6. Rewritten to 3/5, every reference still resolves.
        // Not R-certified: haven reads "" for every cell of the rewritten file.
        let case = Case::load("data_readers_formats_values");
        let mut b = fixture_bytes("dta119_formats.dta");
        let (data, row, offsets) = row_layout(&b);
        let sl = offsets[column_index(&b, "sl")];
        for r in 0..5 {
            let at = data + r * row + sl;
            let v = u64::from(u16::from_le_bytes([b[at], b[at + 1]]));
            let mut o = [0u8; 8];
            o[..6].copy_from_slice(&b[at + 2..at + 8]);
            let o = u64::from_le_bytes(o);
            let packed = v | (o << 24);
            put(&mut b, at, &packed.to_le_bytes());
        }
        let df = load(b).unwrap();
        assert_text_matches(&case, "dta119_formats__sl", df.column("sl").unwrap());
    }

    #[test]
    fn test_stata_strl_respects_the_allocation_ceiling() {
        // The 3,000-byte strL is refused under a 1,024-byte ceiling; the column buffers (7 x 5 x
        // 24 = 840 bytes) are not.
        let e = crate::resource::with_alloc_ceiling(Some(1024), || {
            load_stata(fixture_path("dta118_formats.dta")).unwrap_err()
        });
        match e {
            StataError::ParseError(m) => {
                assert!(m.contains("Stata strL"), "{m}")
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn test_stata_date_with_a_fraction_stays_numeric() {
        let mut b = fixture_bytes("dta118_formats.dta");
        let (data, _, offsets) = row_layout(&b);
        let j = column_index(&b, "dt");
        let reader = StataReader::new(Cursor::new(b.clone())).unwrap();
        assert!(matches!(reader.variables[j].dtype, StataType::Double));
        put(&mut b, data + offsets[j], &0.5f64.to_le_bytes());
        let load = StataReader::new(Cursor::new(b))
            .unwrap()
            .read_with(&StataOptions::default())
            .unwrap();
        let dt = load.df.column("dt").unwrap();
        assert_eq!(dt.dtype(), &DataType::Float64);
        assert_eq!(dt.f64().unwrap().get(0), Some(0.5));
        assert!(
            load.notes
                .iter()
                .any(|n| n.contains("'dt'") && n.contains("%td") && n.contains("loads as numbers")),
            "{:?}",
            load.notes
        );
    }

    #[test]
    fn test_stata_value_labels_option_parses() {
        assert_eq!("text".parse::<ValueLabels>().unwrap(), ValueLabels::Text);
        assert_eq!("Codes".parse::<ValueLabels>().unwrap(), ValueLabels::Codes);
        assert!("labels".parse::<ValueLabels>().is_err());
        assert_eq!(StataOptions::default().value_labels, ValueLabels::Codes);
    }

    // ---- refusals ----

    #[test]
    fn test_validate_stata_k0_n_positive_is_error() {
        // DATA-73: with <K> = 0 the N guard never ran and the row loop spun N times. ReadStat
        // refuses the same patched bytes (case values guard__dta_k0_n3, guard__dta_k0_n2p40 = -1).
        // N = 2^40 comes last: before the fix it spins (debug builds), N = 3 fails first.
        let case = Case::load("data_readers_values");
        assert_eq!(case.scalar("guard__dta_k0_n3"), -1.0);
        assert_eq!(case.scalar("guard__dta_k0_n2p40"), -1.0);
        for n in [3u64, 0, 1 << 40] {
            let mut b = fixture_bytes("dta118_haven.dta");
            put(&mut b, K118, &0u16.to_le_bytes());
            put(&mut b, N118, &n.to_le_bytes());
            assert_invalid(
                load(b),
                &format!("no variables (<K> = 0) and {n} observations"),
            );
        }
    }

    #[test]
    fn test_stata_column_buffers_respect_the_allocation_ceiling() {
        // G3: 6 variables x 6 rows x 24 bytes = 864 bytes of column buffers, refused under a
        // 64-byte per-allocation ceiling instead of allocated.
        let e = crate::resource::with_alloc_ceiling(Some(64), || {
            load_stata(fixture_path("dta118_haven.dta")).unwrap_err()
        });
        match e {
            StataError::ParseError(m) => {
                assert!(
                    m.contains("Stata columns") && m.contains("per-allocation limit"),
                    "{m}"
                )
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
        assert!(load_stata(fixture_path("dta118_haven.dta")).is_ok()); // ceiling restored
    }

    #[test]
    fn test_stata_cell_check_covers_every_declared_row() {
        // The buffers start at 2^20 rows at most, and the rest grows in the read. dta118_str1 has
        // one str1 column, a 1-byte row (same <K>, <N> and <map> offsets as dta118_haven):
        // 1.5 x 2^20 declared rows need 36 MiB of cells, above this 32 MiB ceiling, while the
        // first buffer's 2^20 cells (24 MiB) are not.
        let mut b = fixture_bytes("dta118_str1.dta");
        assert_eq!(&b[K118 - 3..K118], b"<K>");
        assert_eq!(&b[N118 - 3..N118], b"<N>");
        assert_eq!(&b[MAP118 - 5..MAP118], b"<map>");
        assert_eq!(u16::from_le_bytes([b[K118], b[K118 + 1]]), 1);
        let n = 3u64 << 19;
        let end = b.len() + n as usize;
        b.resize(end, 0);
        put(&mut b, N118, &n.to_le_bytes());
        put(&mut b, MAP118 + 8 * 10, &(end as u64).to_le_bytes()); // <strls>, after the rows
        let first_buffer = (1usize << 20) * std::mem::size_of::<Option<String>>();
        let ceiling = 32usize << 20;
        assert!(
            first_buffer < ceiling
                && (n as usize) * std::mem::size_of::<Option<String>>() > ceiling
        );
        let e = crate::resource::with_alloc_ceiling(Some(ceiling), || load(b).unwrap_err());
        match e {
            StataError::ParseError(m) => assert!(m.contains("Stata columns"), "{m}"),
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_read_dta_trailing_blanks() {
        // ReadStat strips a str# value's trailing blanks and keeps its leading ones. `sw` is a
        // str26, 23 bytes into the 57-byte row; the case patches the same cells of the same file.
        let case = Case::load("data_readers_values");
        let mut b = fixture_bytes("dta118_haven.dta");
        let data = u64::from_le_bytes(b[MAP118 + 72..MAP118 + 80].try_into().unwrap()) as usize + 6;
        for (i, cell) in [&b"x  "[..], b"  y ", b"   "].iter().enumerate() {
            let at = data + 57 * i + 23;
            b[at..at + 26].fill(0);
            put(&mut b, at, cell);
        }
        let df = load(b).unwrap();
        let got: Vec<&str> = df
            .column("sw")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(got, case.names("dta118_blanks__sw"));
        assert_eq!(got[..3], ["x", "  y", ""]);
    }

    #[test]
    fn test_stata_map_and_header_guards() {
        // DATA-22: each <map> offset is checked against the file and against the tag found there;
        // the T0 observation guard is bounded by the <data> section.
        let len = fixture_bytes("dta118_haven.dta").len() as u64;
        let mut b = fixture_bytes("dta118_haven.dta");
        put(&mut b, MAP118 + 8 * 9, &(len + 100).to_le_bytes());
        assert_invalid(
            load(b),
            "<map> places <data> at byte 4837, past the end of the 4737-byte file",
        );
        let mut b = fixture_bytes("dta118_haven.dta");
        put(&mut b, MAP118 + 8 * 2, &282u64.to_le_bytes()); // <variable_types> is at 281
        assert_invalid(load(b), "expected <variable_types> at byte 282");
        let mut b = fixture_bytes("dta118_haven.dta");
        put(&mut b, N118, &1_000_000u64.to_le_bytes());
        assert_invalid(
            load(b),
            "declared observation count 1000000 exceeds what the file can hold (6 rows of 57 \
             bytes in a 349-byte <data> section)",
        );
        let mut b = fixture_bytes("dta118_haven.dta");
        put(&mut b, K118, &60_000u16.to_le_bytes());
        assert_invalid(load(b), "<K> = 60000 variables do not fit");
    }

    #[test]
    fn test_stata_pre_117_formats_are_unsupported() {
        // Formats 102-116 are binary (byte 0 the format, byte 1 the byte order); releases after
        // 119 are not read either.
        assert!(matches!(
            load(vec![114, 2, 1, 0, 0, 0, 0, 0]),
            Err(StataError::UnsupportedVersion(114))
        ));
        let mut b = fixture_bytes("dta118_haven.dta");
        b[28..31].copy_from_slice(b"120");
        assert!(matches!(load(b), Err(StataError::UnsupportedVersion(120))));
        // Three digits that do not fit the error's u8, and text that is not a release at all.
        let mut b = fixture_bytes("dta118_haven.dta");
        b[28..31].copy_from_slice(b"300");
        assert_invalid(
            load(b),
            "release 300 is not supported (117, 118 and 119 are)",
        );
        let mut b = fixture_bytes("dta118_haven.dta");
        b[28..31].copy_from_slice(b"+12");
        assert_invalid(load(b), "release \"+12\" is not a number");
    }
}
