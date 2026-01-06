//! Stata DTA file reader (supports versions 117-119).
//!
//! This is a pure Rust implementation for reading Stata .dta files.

use polars::prelude::*;
use polars::frame::column::Column;
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

        if version < 117 || version > 119 {
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

        // Prepare column builders
        let mut columns: Vec<Vec<AnyValue>> = vec![Vec::with_capacity(self.n_obs as usize); self.n_vars];

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

            if buffer.len() >= tag_len {
                if &buffer[buffer.len() - tag_len..] == tag {
                    return Ok(());
                }
            }

            // Prevent unbounded memory growth
            if buffer.len() > 10_000_000 {
                return Err(StataError::InvalidFormat("Tag search exceeded limit".into()));
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

    #[test]
    fn test_stata_type_from_code() {
        assert!(matches!(StataType::from_code(65530), Ok(StataType::Byte)));
        assert!(matches!(StataType::from_code(65526), Ok(StataType::Double)));
        assert!(matches!(StataType::from_code(100), Ok(StataType::Str(100))));
    }
}
