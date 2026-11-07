//! Stata DTA file reader for formats 117 (Stata 13), 118 (Stata 14-18) and 119 (Stata 15-18, more
//! than 32,767 variables), little-endian (LSF) and big-endian (MSF).
//!
//! This is a pure Rust implementation of Stata's `dta` specification. The header is read tag by
//! tag; the `<map>` then locates every section (slot 2 `<variable_types>`, 3 `<varnames>`, 9
//! `<data>`, 10 `<strls>`), and the reader checks the tag at each offset instead of scanning for
//! it. Field widths by release:
//!
//! | Release | `<K>` | `<N>` | label length | variable name |
//! |---|---|---|---|---|
//! | 117 | u16 | u32 | u8 | 33 bytes |
//! | 118 | u16 | u64 | u16 | 129 bytes |
//! | 119 | u32 | u64 | u16 | 129 bytes |
//!
//! Text is UTF-8 in releases 118 and 119. Release 117 text is read as UTF-8 when it is valid UTF-8
//! (haven and ReadStat write UTF-8) and as Windows-1252 otherwise (Stata 13 writes the platform
//! encoding). Values are pinned to `haven::read_dta` on the fixtures in
//! `validation/datasets/files/` (`validation/data/readers.md`). Formats before 117 (Stata 12 and
//! earlier) are refused with [`StataError::UnsupportedVersion`]; strL contents and value labels are
//! not decoded (see the strL warning in [`StataReader::read_to_dataframe`]).

use polars::frame::column::Column;
use polars::prelude::*;
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
const MAP_DATA: usize = 9;
const MAP_STRLS: usize = 10;
const DATA_TAG: &[u8] = b"<data>";
const DATA_END_TAG: &[u8] = b"</data>";

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
        let n_vars = usize::try_from(n_vars_raw)
            .map_err(|_| StataError::InvalidFormat(format!("<K> = {n_vars_raw} is too large")))?;

        Self::seek_tag(&mut reader, map[MAP_VARIABLE_TYPES], b"<variable_types>")?;
        let mut var_types = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            let code = Self::read_uint(&mut reader, 2, byte_order)? as u16;
            var_types.push(StataType::from_code(code)?);
        }

        Self::seek_tag(&mut reader, map[MAP_VARNAMES], b"<varnames>")?;
        let mut variables = Vec::with_capacity(n_vars);
        for (i, dtype) in var_types.into_iter().enumerate() {
            let name = Self::read_fixed_string(&mut reader, release.name_len(), release, || {
                format!("the name of variable {}", i + 1)
            })?;
            variables.push(Variable { name, dtype });
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
        })
    }

    /// Read the entire dataset into a Polars DataFrame.
    pub fn read_to_dataframe(mut self) -> Result<DataFrame, StataError> {
        self.reader.seek(SeekFrom::Start(self.data_offset))?;

        // Clone variable info to avoid borrow issues
        let var_types: Vec<StataType> = self.variables.iter().map(|v| v.dtype.clone()).collect();
        let var_names: Vec<String> = self.variables.iter().map(|v| v.name.clone()).collect();

        // strL columns are not yet decoded (the strls section is skipped); each cell
        // is replaced with the literal placeholder "<strL>". Surface this clearly so
        // the caller does not silently consume corrupted string data.
        let strl_columns: Vec<&String> = var_types
            .iter()
            .zip(&var_names)
            .filter_map(|(dtype, name)| matches!(dtype, StataType::StrL).then_some(name))
            .collect();
        if !strl_columns.is_empty() {
            tracing::warn!(
                "Stata file contains long-string (strL) columns whose contents are not yet \
                 supported by this reader; values in column(s) {:?} are replaced with the \
                 placeholder \"<strL>\"",
                strl_columns
            );
        }

        // Guard against a corrupt/malicious header declaring an implausible
        // observation count. `n_obs` comes straight from the file's `<N>` tag;
        // using it to pre-size the column buffers below would let a tiny file
        // trigger a multi-gigabyte allocation. The `<data>` section, located by
        // `<map>`, holds exactly the rows and `</data>`.
        let row_size: u64 = var_types.iter().map(|t| t.size() as u64).sum();
        let rows_end = DATA_END_TAG.len() as u64;
        let max_obs = if row_size == 0 {
            0
        } else {
            self.data_len.saturating_sub(rows_end) / row_size
        };
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

        // Prepare column builders. Capacity is bounded above (see the guard
        // just above), so this cannot be abused for a huge allocation.
        let capacity = (self.n_obs as usize).min(1 << 20);
        let mut columns: Vec<Vec<AnyValue>> = vec![Vec::with_capacity(capacity); self.n_vars];

        // Read observations
        for _ in 0..self.n_obs {
            for (var_idx, dtype) in var_types.iter().enumerate() {
                let value = self.read_value(dtype, &var_names[var_idx])?;
                columns[var_idx].push(value);
            }
        }

        // Convert to Polars columns
        let mut polars_columns: Vec<Column> = Vec::with_capacity(self.n_vars);

        for (var_idx, dtype) in var_types.iter().enumerate() {
            let col = self.build_column(&var_names[var_idx], dtype, &columns[var_idx])?;
            polars_columns.push(col);
        }

        DataFrame::new(polars_columns).map_err(StataError::PolarsError)
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
                return Err(
                    match std::str::from_utf8(other)
                        .ok()
                        .and_then(|s| s.parse::<u8>().ok())
                    {
                        Some(v) => StataError::UnsupportedVersion(v),
                        None => StataError::InvalidFormat(format!(
                            "release {:?} is not a number",
                            decode_windows_1252_lenient(other)
                        )),
                    },
                );
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

    /// A NUL-terminated fixed-width string, decoded by the release's text rule.
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
        release.decode(&buf[..end], what)
    }
    fn read_value(
        &mut self,
        dtype: &StataType,
        name: &str,
    ) -> Result<AnyValue<'static>, StataError> {
        match dtype {
            StataType::Byte => {
                let v = Self::read_i8(&mut self.reader)?;
                // Stata missing values for byte: > 100
                if v > 100 {
                    Ok(AnyValue::Null)
                } else {
                    Ok(AnyValue::Int64(v as i64))
                }
            }
            StataType::Int => {
                let v = Self::read_i16(&mut self.reader, self.byte_order)?;
                // Stata missing values for int: > 32740
                if v > 32740 {
                    Ok(AnyValue::Null)
                } else {
                    Ok(AnyValue::Int64(v as i64))
                }
            }
            StataType::Long => {
                let v = Self::read_i32(&mut self.reader, self.byte_order)?;
                // Stata missing values for long: > 2147483620
                if v > 2147483620 {
                    Ok(AnyValue::Null)
                } else {
                    Ok(AnyValue::Int64(v as i64))
                }
            }
            StataType::Float => {
                let v = Self::read_f32(&mut self.reader, self.byte_order)?;
                if v.is_nan() || v > 1.701e38 {
                    Ok(AnyValue::Null)
                } else {
                    Ok(AnyValue::Float64(v as f64))
                }
            }
            StataType::Double => {
                let v = Self::read_f64(&mut self.reader, self.byte_order)?;
                if v.is_nan() || v > 8.988e307 {
                    Ok(AnyValue::Null)
                } else {
                    Ok(AnyValue::Float64(v))
                }
            }
            StataType::Str(len) => {
                let release = self.release;
                let s = Self::read_fixed_string(&mut self.reader, *len, release, || {
                    format!("a value of variable {name}")
                })?;
                Ok(AnyValue::StringOwned(s.into()))
            }
            StataType::StrL => {
                // Read strL reference (v, o) - 8 bytes
                let mut buf = [0u8; 8];
                self.reader.read_exact(&mut buf)?;
                // For now, return placeholder - full strL support requires reading strls section
                Ok(AnyValue::StringOwned("<strL>".into()))
            }
        }
    }

    fn build_column(
        &self,
        name: &str,
        dtype: &StataType,
        values: &[AnyValue<'static>],
    ) -> Result<Column, StataError> {
        match dtype {
            StataType::Byte | StataType::Int | StataType::Long => {
                let vals: Vec<Option<i64>> = values
                    .iter()
                    .map(|v| match v {
                        AnyValue::Int64(i) => Some(*i),
                        _ => None,
                    })
                    .collect();
                Ok(Column::new(name.into(), vals))
            }
            StataType::Float | StataType::Double => {
                let vals: Vec<Option<f64>> = values
                    .iter()
                    .map(|v| match v {
                        AnyValue::Float64(f) => Some(*f),
                        _ => None,
                    })
                    .collect();
                Ok(Column::new(name.into(), vals))
            }
            StataType::Str(_) | StataType::StrL => {
                let vals: Vec<Option<String>> = values
                    .iter()
                    .map(|v| match v {
                        AnyValue::StringOwned(s) => Some(s.to_string()),
                        _ => None,
                    })
                    .collect();
                Ok(Column::new(name.into(), vals))
            }
        }
    }
}

/// Load a Stata DTA file into a Polars DataFrame.
pub fn load_stata(path: impl AsRef<Path>) -> Result<DataFrame, StataError> {
    let file = std::fs::File::open(path)?;
    let reader = StataReader::new(file)?;
    reader.read_to_dataframe()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::sas::fixture_check::{assert_matches_haven, fixture_bytes, fixture_path, put};
    use p2a_validation::Case;
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
    fn test_validate_dataloader_loads_dta_fixture() {
        // The path `p2a data load` and MCP load_dataset take.
        let case = Case::load("data_readers_values");
        let ds = crate::data::DataLoader::load(fixture_path("dta118_haven.dta")).unwrap();
        assert_eq!(ds.name(), Some("dta118_haven"));
        assert_matches_haven(&case, "dta118_haven", ds.df());
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
    }
}
