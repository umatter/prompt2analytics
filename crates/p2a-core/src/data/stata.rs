//! Stata DTA file reader (supports versions 117-119).
//!
//! This is a pure Rust implementation for reading Stata .dta files.

use polars::frame::column::Column;
use polars::prelude::*;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
    #[allow(dead_code)]
    version: u8,
    n_vars: usize,
    n_obs: u64,
    variables: Vec<Variable>,
    data_offset: u64,
}

impl<R: Read + Seek> StataReader<R> {
    /// Create a new Stata reader from a file.
    pub fn new(reader: R) -> Result<Self, StataError> {
        let mut reader = BufReader::new(reader);

        // Read and parse header
        let (version, byte_order) = Self::read_header(&mut reader)?;

        if !(117..=119).contains(&version) {
            return Err(StataError::UnsupportedVersion(version));
        }

        // Skip to map section and read offsets
        Self::skip_to_tag(&mut reader, b"<map>")?;
        let offsets = Self::read_map(&mut reader, byte_order)?;

        // Read number of variables and observations
        reader.seek(SeekFrom::Start(offsets[1]))?; // variable_types offset
        Self::skip_to_tag(&mut reader, b"<K>")?;
        let n_vars = Self::read_u16(&mut reader, byte_order)? as usize;

        Self::skip_to_tag(&mut reader, b"<N>")?;
        let n_obs = Self::read_u64(&mut reader, byte_order)?;

        // Read variable types
        Self::skip_to_tag(&mut reader, b"<variable_types>")?;
        let mut var_types = Vec::with_capacity(n_vars);
        for _ in 0..n_vars {
            let code = Self::read_u16(&mut reader, byte_order)?;
            var_types.push(StataType::from_code(code)?);
        }

        // Read variable names
        Self::skip_to_tag(&mut reader, b"<varnames>")?;
        let mut variables = Vec::with_capacity(n_vars);
        let name_len = if version >= 118 { 129 } else { 33 };

        for i in 0..n_vars {
            let name = Self::read_fixed_string(&mut reader, name_len)?;
            variables.push(Variable {
                name,
                dtype: var_types[i].clone(),
            });
        }

        // Find data offset
        let data_offset = offsets[5]; // data section offset

        Ok(StataReader {
            reader,
            byte_order,
            version,
            n_vars,
            n_obs,
            variables,
            data_offset,
        })
    }

    /// Read the entire dataset into a Polars DataFrame.
    pub fn read_to_dataframe(mut self) -> Result<DataFrame, StataError> {
        // Seek to data section
        self.reader.seek(SeekFrom::Start(self.data_offset))?;
        Self::skip_to_tag(&mut self.reader, b"<data>")?;

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

        // Prepare column builders
        let mut columns: Vec<Vec<AnyValue>> =
            vec![Vec::with_capacity(self.n_obs as usize); self.n_vars];

        // Read observations
        for _ in 0..self.n_obs {
            for (var_idx, dtype) in var_types.iter().enumerate() {
                let value = self.read_value(dtype)?;
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

    fn read_header(reader: &mut BufReader<R>) -> Result<(u8, ByteOrder), StataError> {
        // Read until we find <stata_dta>
        Self::skip_to_tag(reader, b"<stata_dta>")?;

        // Read <header>
        Self::skip_to_tag(reader, b"<release>")?;

        // Read version number (3 bytes as ASCII)
        let mut version_buf = [0u8; 3];
        reader.read_exact(&mut version_buf)?;
        let version_str = std::str::from_utf8(&version_buf)
            .map_err(|_| StataError::InvalidFormat("Invalid version string".into()))?;
        let version: u8 = version_str
            .trim()
            .parse()
            .map_err(|_| StataError::InvalidFormat("Cannot parse version".into()))?;

        // Skip to byteorder
        Self::skip_to_tag(reader, b"<byteorder>")?;
        let mut order_buf = [0u8; 3];
        reader.read_exact(&mut order_buf)?;

        let byte_order = match &order_buf {
            b"LSF" => ByteOrder::LittleEndian,
            b"MSF" => ByteOrder::BigEndian,
            _ => return Err(StataError::InvalidFormat("Invalid byte order".into())),
        };

        Ok((version, byte_order))
    }

    fn skip_to_tag(reader: &mut BufReader<R>, tag: &[u8]) -> Result<(), StataError> {
        let mut buffer = Vec::new();
        let tag_len = tag.len();

        loop {
            let byte = {
                let mut b = [0u8; 1];
                if reader.read(&mut b)? == 0 {
                    return Err(StataError::InvalidFormat(format!(
                        "Tag not found: {}",
                        String::from_utf8_lossy(tag)
                    )));
                }
                b[0]
            };

            buffer.push(byte);

            if buffer.len() >= tag_len && &buffer[buffer.len() - tag_len..] == tag {
                return Ok(());
            }

            // Prevent unbounded memory growth
            if buffer.len() > 10_000_000 {
                return Err(StataError::InvalidFormat(
                    "Tag search exceeded limit".into(),
                ));
            }
        }
    }

    fn read_map(reader: &mut BufReader<R>, byte_order: ByteOrder) -> Result<Vec<u64>, StataError> {
        let mut offsets = Vec::with_capacity(14);
        for _ in 0..14 {
            offsets.push(Self::read_u64(reader, byte_order)?);
        }
        Ok(offsets)
    }

    fn read_u16(reader: &mut BufReader<R>, byte_order: ByteOrder) -> Result<u16, StataError> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => u16::from_le_bytes(buf),
            ByteOrder::BigEndian => u16::from_be_bytes(buf),
        })
    }

    fn read_u64(reader: &mut BufReader<R>, byte_order: ByteOrder) -> Result<u64, StataError> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => u64::from_le_bytes(buf),
            ByteOrder::BigEndian => u64::from_be_bytes(buf),
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

    fn read_fixed_string(reader: &mut BufReader<R>, len: usize) -> Result<String, StataError> {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;

        // Find null terminator
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());

        String::from_utf8(buf[..end].to_vec())
            .map_err(|_| StataError::ParseError("Invalid UTF-8 in string".into()))
    }

    fn read_value(&mut self, dtype: &StataType) -> Result<AnyValue<'static>, StataError> {
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
                let s = Self::read_fixed_string(&mut self.reader, *len)?;
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
    use std::io::Cursor;

    #[test]
    fn test_stata_type_from_code() {
        assert!(matches!(StataType::from_code(65530), Ok(StataType::Byte)));
        assert!(matches!(StataType::from_code(65526), Ok(StataType::Double)));
        assert!(matches!(StataType::from_code(100), Ok(StataType::Str(100))));
    }

    /// Generate a minimal valid Stata DTA version 118 file.
    fn create_test_dta_file() -> Vec<u8> {
        let mut buf = Vec::new();

        // === Header section ===
        buf.extend_from_slice(b"<stata_dta>");
        buf.extend_from_slice(b"<header>");
        buf.extend_from_slice(b"<release>118</release>");
        buf.extend_from_slice(b"<byteorder>LSF</byteorder>"); // Little-endian
        // Track position BEFORE <K> - the reader seeks to offsets[1] then scans for <K>
        let before_k_offset = buf.len() as u64;
        buf.extend_from_slice(b"<K>");
        buf.extend_from_slice(&2u16.to_le_bytes()); // 2 variables
        buf.extend_from_slice(b"</K>");
        buf.extend_from_slice(b"<N>");
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 observations
        buf.extend_from_slice(b"</N>");
        buf.extend_from_slice(b"<label>");
        buf.extend_from_slice(&0u16.to_le_bytes()); // No label
        buf.extend_from_slice(b"</label>");
        buf.extend_from_slice(b"<timestamp>");
        buf.extend_from_slice(&0u8.to_le_bytes()); // No timestamp
        buf.extend_from_slice(b"</timestamp>");
        buf.extend_from_slice(b"</header>");

        // === Map section (14 offsets) ===
        buf.extend_from_slice(b"<map>");

        // Placeholder offsets - will be filled in
        let offset_placeholder_pos = buf.len();
        for _ in 0..14 {
            buf.extend_from_slice(&0u64.to_le_bytes());
        }
        buf.extend_from_slice(b"</map>");

        // === Variable types section ===
        let var_types_offset = buf.len() as u64;
        buf.extend_from_slice(b"<variable_types>");
        buf.extend_from_slice(&65526u16.to_le_bytes()); // Double
        buf.extend_from_slice(&65526u16.to_le_bytes()); // Double
        buf.extend_from_slice(b"</variable_types>");

        // === Variable names section ===
        let _varnames_offset = buf.len() as u64;
        buf.extend_from_slice(b"<varnames>");
        // Version 118 uses 129-byte names
        let mut name1 = vec![0u8; 129];
        name1[..2].copy_from_slice(b"x1");
        buf.extend_from_slice(&name1);
        let mut name2 = vec![0u8; 129];
        name2[..2].copy_from_slice(b"x2");
        buf.extend_from_slice(&name2);
        buf.extend_from_slice(b"</varnames>");

        // === Sortlist section ===
        buf.extend_from_slice(b"<sortlist>");
        buf.extend_from_slice(
            &[0u16; 3]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        buf.extend_from_slice(b"</sortlist>");

        // === Formats section ===
        buf.extend_from_slice(b"<formats>");
        // 57-byte format strings for version 118
        let mut fmt1 = vec![0u8; 57];
        fmt1[..6].copy_from_slice(b"%10.0g");
        buf.extend_from_slice(&fmt1);
        let mut fmt2 = vec![0u8; 57];
        fmt2[..6].copy_from_slice(b"%10.0g");
        buf.extend_from_slice(&fmt2);
        buf.extend_from_slice(b"</formats>");

        // === Value label names section ===
        buf.extend_from_slice(b"<value_label_names>");
        // 129-byte value label names
        buf.extend_from_slice(&vec![0u8; 129 * 2]);
        buf.extend_from_slice(b"</value_label_names>");

        // === Variable labels section ===
        buf.extend_from_slice(b"<variable_labels>");
        // 321-byte variable labels for version 118
        buf.extend_from_slice(&vec![0u8; 321 * 2]);
        buf.extend_from_slice(b"</variable_labels>");

        // === Characteristics section ===
        buf.extend_from_slice(b"<characteristics>");
        buf.extend_from_slice(b"</characteristics>");

        // === Data section ===
        let data_offset = buf.len() as u64;
        buf.extend_from_slice(b"<data>");
        // 3 observations, 2 doubles each
        // Row 1: 1.0, 2.0
        buf.extend_from_slice(&1.0f64.to_le_bytes());
        buf.extend_from_slice(&2.0f64.to_le_bytes());
        // Row 2: 3.0, 4.0
        buf.extend_from_slice(&3.0f64.to_le_bytes());
        buf.extend_from_slice(&4.0f64.to_le_bytes());
        // Row 3: 5.0, 6.0
        buf.extend_from_slice(&5.0f64.to_le_bytes());
        buf.extend_from_slice(&6.0f64.to_le_bytes());
        buf.extend_from_slice(b"</data>");

        // === Strls section (empty) ===
        buf.extend_from_slice(b"<strls>");
        buf.extend_from_slice(b"</strls>");

        // === Value labels section (empty) ===
        buf.extend_from_slice(b"<value_labels>");
        buf.extend_from_slice(b"</value_labels>");

        // === End ===
        buf.extend_from_slice(b"</stata_dta>");

        // Now go back and fill in the map offsets
        // The reader: seeks to offsets[1], scans forward for <K>, then <N>, then <variable_types>, etc.
        // offsets[5] is used to seek before scanning for <data>
        let offsets: [u64; 14] = [
            0,                // 0: stata_dta
            before_k_offset,  // 1: position before <K> tag
            var_types_offset, // 2: variable_types (unused by reader)
            0,                // 3: sortlist
            0,                // 4: formats
            data_offset,      // 5: data - reader seeks here, then scans for <data>
            0,                // 6: strls
            0,                // 7: value_labels
            0,
            0,
            0,
            0,
            0,                // 8-12: reserved
            buf.len() as u64, // 13: end of file
        ];

        // Write offsets to the map section
        let mut offset_buf = Vec::new();
        for offset in &offsets {
            offset_buf.extend_from_slice(&offset.to_le_bytes());
        }
        buf[offset_placeholder_pos..offset_placeholder_pos + 112].copy_from_slice(&offset_buf);

        buf
    }

    #[test]
    fn test_read_synthetic_dta() {
        let dta_bytes = create_test_dta_file();
        let cursor = Cursor::new(dta_bytes);
        let reader = StataReader::new(cursor).expect("Failed to create reader");

        assert_eq!(reader.n_vars, 2);
        assert_eq!(reader.n_obs, 3);
        assert_eq!(reader.variables.len(), 2);
        assert_eq!(reader.variables[0].name, "x1");
        assert_eq!(reader.variables[1].name, "x2");

        let df = reader
            .read_to_dataframe()
            .expect("Failed to read dataframe");

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 2);

        // Check values
        let x1 = df.column("x1").unwrap().f64().unwrap();
        let x2 = df.column("x2").unwrap().f64().unwrap();

        assert_eq!(x1.get(0), Some(1.0));
        assert_eq!(x1.get(1), Some(3.0));
        assert_eq!(x1.get(2), Some(5.0));
        assert_eq!(x2.get(0), Some(2.0));
        assert_eq!(x2.get(1), Some(4.0));
        assert_eq!(x2.get(2), Some(6.0));
    }

    #[test]
    fn test_stata_invalid_magic() {
        let invalid_data = b"Not a valid Stata file";
        let cursor = Cursor::new(invalid_data.to_vec());
        let result = StataReader::new(cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_stata_missing_values() {
        // Create a DTA file with missing values
        let mut buf = Vec::new();

        buf.extend_from_slice(b"<stata_dta>");
        buf.extend_from_slice(b"<header>");
        buf.extend_from_slice(b"<release>118</release>");
        buf.extend_from_slice(b"<byteorder>LSF</byteorder>");
        // Track position before <K>
        let before_k_offset = buf.len() as u64;
        buf.extend_from_slice(b"<K>");
        buf.extend_from_slice(&1u16.to_le_bytes()); // 1 variable
        buf.extend_from_slice(b"</K>");
        buf.extend_from_slice(b"<N>");
        buf.extend_from_slice(&2u64.to_le_bytes()); // 2 observations
        buf.extend_from_slice(b"</N>");
        buf.extend_from_slice(b"<label>");
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(b"</label>");
        buf.extend_from_slice(b"<timestamp>");
        buf.extend_from_slice(&0u8.to_le_bytes());
        buf.extend_from_slice(b"</timestamp>");
        buf.extend_from_slice(b"</header>");

        // Map
        buf.extend_from_slice(b"<map>");
        let offset_pos = buf.len();
        for _ in 0..14 {
            buf.extend_from_slice(&0u64.to_le_bytes());
        }
        buf.extend_from_slice(b"</map>");

        // Variable types
        let var_types_offset = buf.len() as u64;
        buf.extend_from_slice(b"<variable_types>");
        buf.extend_from_slice(&65526u16.to_le_bytes()); // Double
        buf.extend_from_slice(b"</variable_types>");

        // Variable names
        buf.extend_from_slice(b"<varnames>");
        let mut name = vec![0u8; 129];
        name[..3].copy_from_slice(b"val");
        buf.extend_from_slice(&name);
        buf.extend_from_slice(b"</varnames>");

        // Sortlist
        buf.extend_from_slice(b"<sortlist>");
        buf.extend_from_slice(
            &[0u16; 2]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        buf.extend_from_slice(b"</sortlist>");

        // Formats
        buf.extend_from_slice(b"<formats>");
        let mut fmt = vec![0u8; 57];
        fmt[..6].copy_from_slice(b"%10.0g");
        buf.extend_from_slice(&fmt);
        buf.extend_from_slice(b"</formats>");

        // Value label names
        buf.extend_from_slice(b"<value_label_names>");
        buf.extend_from_slice(&[0u8; 129]);
        buf.extend_from_slice(b"</value_label_names>");

        // Variable labels
        buf.extend_from_slice(b"<variable_labels>");
        buf.extend_from_slice(&vec![0u8; 321]);
        buf.extend_from_slice(b"</variable_labels>");

        // Characteristics
        buf.extend_from_slice(b"<characteristics>");
        buf.extend_from_slice(b"</characteristics>");

        // Data
        let data_offset = buf.len() as u64;
        buf.extend_from_slice(b"<data>");
        // Row 1: valid value
        buf.extend_from_slice(&42.0f64.to_le_bytes());
        // Row 2: Stata missing value (NaN pattern > 8.988e307)
        buf.extend_from_slice(&f64::NAN.to_le_bytes());
        buf.extend_from_slice(b"</data>");

        // Strls
        buf.extend_from_slice(b"<strls>");
        buf.extend_from_slice(b"</strls>");

        // Value labels
        buf.extend_from_slice(b"<value_labels>");
        buf.extend_from_slice(b"</value_labels>");

        buf.extend_from_slice(b"</stata_dta>");

        // Fill map - offsets[1] must point BEFORE <K>
        let offsets: [u64; 14] = [
            0,
            before_k_offset,
            var_types_offset,
            0,
            0,
            data_offset,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            buf.len() as u64,
        ];
        let mut offset_buf = Vec::new();
        for offset in &offsets {
            offset_buf.extend_from_slice(&offset.to_le_bytes());
        }
        buf[offset_pos..offset_pos + 112].copy_from_slice(&offset_buf);

        let cursor = Cursor::new(buf);
        let reader = StataReader::new(cursor).expect("Failed to create reader");
        let df = reader
            .read_to_dataframe()
            .expect("Failed to read dataframe");

        let col = df.column("val").unwrap().f64().unwrap();
        assert_eq!(col.get(0), Some(42.0));
        assert_eq!(col.get(1), None); // Missing value should be null
    }
}
