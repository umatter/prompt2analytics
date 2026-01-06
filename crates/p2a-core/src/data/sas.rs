//! SAS7BDAT file reader (pure Rust implementation).
//!
//! Supports reading SAS data files (.sas7bdat) into Polars DataFrames.

use polars::prelude::*;
use polars::frame::column::Column;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

/// Magic number for SAS7BDAT files (32 bytes)
const SAS_MAGIC: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xc2, 0xea, 0x81, 0x60,
    0xb3, 0x14, 0x11, 0xcf, 0xbd, 0x92, 0x08, 0x00,
    0x09, 0xc7, 0x31, 0x8c, 0x18, 0x1f, 0x10, 0x11,
];

/// Subheader signatures
const ROW_SIZE_SIGNATURE: [u8; 4] = [0xF7, 0xF7, 0xF7, 0xF7];
const COL_SIZE_SIGNATURE: [u8; 4] = [0xF6, 0xF6, 0xF6, 0xF6];
const COL_TEXT_SIGNATURE: [u8; 4] = [0xFD, 0xFF, 0xFF, 0xFF];
const COL_ATTR_SIGNATURE: [u8; 4] = [0xFC, 0xFF, 0xFF, 0xFF];
const COL_NAME_SIGNATURE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];

/// Errors that can occur during SAS file reading.
#[derive(Error, Debug)]
pub enum SasError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid SAS file: {0}")]
    InvalidFormat(String),

    #[error("Unsupported compression: {0}")]
    UnsupportedCompression(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),
}

/// Byte order for the file
#[derive(Debug, Clone, Copy, PartialEq)]
enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// Column type
#[derive(Debug, Clone, Copy, PartialEq)]
enum ColumnType {
    Numeric,
    Character,
}

/// Column metadata
#[derive(Debug, Clone)]
struct ColumnInfo {
    name: String,
    col_type: ColumnType,
    offset: usize,
    length: usize,
}

/// SAS7BDAT file reader
pub struct SasReader<R: Read + Seek> {
    reader: R,
    byte_order: ByteOrder,
    is_64bit: bool,
    header_length: usize,
    page_size: usize,
    page_count: usize,
    row_length: usize,
    row_count: usize,
    columns: Vec<ColumnInfo>,
}

impl<R: Read + Seek> SasReader<R> {
    /// Create a new SAS reader.
    pub fn new(mut reader: R) -> Result<Self, SasError> {
        // Verify magic number
        let mut magic = [0u8; 32];
        reader.read_exact(&mut magic)?;
        if magic != SAS_MAGIC {
            return Err(SasError::InvalidFormat("Invalid SAS7BDAT magic number".into()));
        }

        // Read alignment and platform info
        reader.seek(SeekFrom::Start(32))?;
        let mut align_byte = [0u8; 1];
        reader.read_exact(&mut align_byte)?;
        let is_64bit = align_byte[0] == 0x33;

        reader.seek(SeekFrom::Start(35))?;
        let mut align2_byte = [0u8; 1];
        reader.read_exact(&mut align2_byte)?;

        reader.seek(SeekFrom::Start(37))?;
        let mut endian_byte = [0u8; 1];
        reader.read_exact(&mut endian_byte)?;
        let byte_order = if endian_byte[0] == 0x01 {
            ByteOrder::LittleEndian
        } else {
            ByteOrder::BigEndian
        };

        // Calculate alignment offsets
        let a1 = if is_64bit { 4 } else { 0 };
        let _a2 = if align2_byte[0] == 0x33 { 4 } else { 0 };

        // Read header length
        reader.seek(SeekFrom::Start((196 + a1) as u64))?;
        let header_length = Self::read_i32(&mut reader, byte_order)? as usize;

        // Read page size
        reader.seek(SeekFrom::Start((200 + a1) as u64))?;
        let page_size = Self::read_i32(&mut reader, byte_order)? as usize;

        // Read page count
        reader.seek(SeekFrom::Start((204 + a1) as u64))?;
        let page_count = if is_64bit {
            Self::read_i64(&mut reader, byte_order)? as usize
        } else {
            Self::read_i32(&mut reader, byte_order)? as usize
        };

        let mut sas_reader = SasReader {
            reader,
            byte_order,
            is_64bit,
            header_length,
            page_size,
            page_count,
            row_length: 0,
            row_count: 0,
            columns: Vec::new(),
        };

        // Parse metadata pages
        sas_reader.parse_metadata()?;

        Ok(sas_reader)
    }

    /// Parse metadata from pages to get column info.
    fn parse_metadata(&mut self) -> Result<(), SasError> {
        let _int_size = if self.is_64bit { 8 } else { 4 };
        let subheader_ptr_size = if self.is_64bit { 24 } else { 12 };
        let page_header_size = if self.is_64bit { 40 } else { 24 };

        let mut col_names: Vec<String> = Vec::new();
        let mut col_attrs: Vec<(usize, usize, ColumnType)> = Vec::new(); // (offset, length, type)

        for page_idx in 0..self.page_count {
            let page_offset = self.header_length + page_idx * self.page_size;
            self.reader.seek(SeekFrom::Start(page_offset as u64))?;

            // Read page type
            let page_type_offset = if self.is_64bit { 32 } else { 16 };
            self.reader.seek(SeekFrom::Start((page_offset + page_type_offset) as u64))?;
            let page_type = Self::read_i16(&mut self.reader, self.byte_order)?;

            // Only process metadata pages (0) and mixed pages (512)
            if page_type != 0 && page_type != 512 {
                continue;
            }

            // Read subheader count
            let sh_count_offset = if self.is_64bit { 34 } else { 18 };
            self.reader.seek(SeekFrom::Start((page_offset + sh_count_offset) as u64))?;
            let subheader_count = Self::read_i16(&mut self.reader, self.byte_order)? as usize;

            // Process subheaders
            for sh_idx in 0..subheader_count {
                let sh_ptr_offset = page_offset + page_header_size + sh_idx * subheader_ptr_size;
                self.reader.seek(SeekFrom::Start(sh_ptr_offset as u64))?;

                // Read subheader offset and length
                let sh_offset = if self.is_64bit {
                    Self::read_i64(&mut self.reader, self.byte_order)? as usize
                } else {
                    Self::read_i32(&mut self.reader, self.byte_order)? as usize
                };

                let sh_length = if self.is_64bit {
                    Self::read_i64(&mut self.reader, self.byte_order)? as usize
                } else {
                    Self::read_i32(&mut self.reader, self.byte_order)? as usize
                };

                if sh_length == 0 {
                    continue;
                }

                // Read subheader signature
                let sh_abs_offset = page_offset + sh_offset;
                self.reader.seek(SeekFrom::Start(sh_abs_offset as u64))?;
                let mut signature = [0u8; 4];
                self.reader.read_exact(&mut signature)?;

                // Process based on signature
                if signature == ROW_SIZE_SIGNATURE {
                    // Row size subheader
                    let row_len_offset = if self.is_64bit { 40 } else { 20 };
                    self.reader.seek(SeekFrom::Start((sh_abs_offset + row_len_offset) as u64))?;
                    self.row_length = if self.is_64bit {
                        Self::read_i64(&mut self.reader, self.byte_order)? as usize
                    } else {
                        Self::read_i32(&mut self.reader, self.byte_order)? as usize
                    };

                    let row_count_offset = if self.is_64bit { 48 } else { 24 };
                    self.reader.seek(SeekFrom::Start((sh_abs_offset + row_count_offset) as u64))?;
                    self.row_count = if self.is_64bit {
                        Self::read_i64(&mut self.reader, self.byte_order)? as usize
                    } else {
                        Self::read_i32(&mut self.reader, self.byte_order)? as usize
                    };
                } else if signature == COL_SIZE_SIGNATURE {
                    // Column size subheader - we get column count here but use attrs
                } else if signature == COL_NAME_SIGNATURE {
                    // Column name subheader
                    self.parse_column_names(sh_abs_offset, sh_length, &mut col_names)?;
                } else if signature == COL_ATTR_SIGNATURE {
                    // Column attributes subheader
                    self.parse_column_attributes(sh_abs_offset, sh_length, &mut col_attrs)?;
                }
            }
        }

        // Combine names and attributes
        for (i, (offset, length, col_type)) in col_attrs.iter().enumerate() {
            let name = col_names.get(i).cloned().unwrap_or_else(|| format!("col_{}", i));
            self.columns.push(ColumnInfo {
                name,
                col_type: *col_type,
                offset: *offset,
                length: *length,
            });
        }

        Ok(())
    }

    /// Parse column names from subheader.
    fn parse_column_names(&mut self, sh_offset: usize, sh_length: usize, names: &mut Vec<String>) -> Result<(), SasError> {
        let int_size = if self.is_64bit { 8 } else { 4 };
        let base_offset = sh_offset + 8 + int_size * 2;

        // Read column name pointers
        let _ptr_size = if self.is_64bit { 8 } else { 4 };
        let entry_size = 8; // Each name entry is 8 bytes

        let remaining = sh_length.saturating_sub(8 + int_size * 2);
        let entry_count = remaining / entry_size;

        for i in 0..entry_count {
            let entry_offset = base_offset + i * entry_size;
            self.reader.seek(SeekFrom::Start(entry_offset as u64))?;

            // Read text index, offset, and length
            let _text_idx = Self::read_i16(&mut self.reader, self.byte_order)?;
            let _name_offset = Self::read_i16(&mut self.reader, self.byte_order)? as usize;
            let name_length = Self::read_i16(&mut self.reader, self.byte_order)? as usize;

            if name_length > 0 && name_length < 256 {
                // Read from column text subheader (simplified - assumes single text block)
                // For now, generate placeholder names
                names.push(format!("VAR{}", i + 1));
            }
        }

        Ok(())
    }

    /// Parse column attributes from subheader.
    fn parse_column_attributes(&mut self, sh_offset: usize, sh_length: usize, attrs: &mut Vec<(usize, usize, ColumnType)>) -> Result<(), SasError> {
        let _int_size = if self.is_64bit { 8 } else { 4 };
        let base_offset = sh_offset + if self.is_64bit { 16 } else { 12 };

        let attr_size = if self.is_64bit { 16 } else { 12 };
        let remaining = sh_length.saturating_sub(if self.is_64bit { 16 } else { 12 });
        let attr_count = remaining / attr_size;

        for i in 0..attr_count {
            let attr_offset = base_offset + i * attr_size;
            self.reader.seek(SeekFrom::Start(attr_offset as u64))?;

            // Read offset within row
            let col_offset = if self.is_64bit {
                Self::read_i64(&mut self.reader, self.byte_order)? as usize
            } else {
                Self::read_i32(&mut self.reader, self.byte_order)? as usize
            };

            // Read column width
            let col_width = Self::read_i32(&mut self.reader, self.byte_order)? as usize;

            // Read column type (1=numeric, 2=character)
            let type_byte = {
                let mut b = [0u8; 1];
                self.reader.read_exact(&mut b)?;
                b[0]
            };

            let col_type = if type_byte == 1 {
                ColumnType::Numeric
            } else {
                ColumnType::Character
            };

            attrs.push((col_offset, col_width, col_type));
        }

        Ok(())
    }

    /// Read the data into a DataFrame.
    pub fn read_to_dataframe(mut self) -> Result<DataFrame, SasError> {
        if self.columns.is_empty() {
            return Err(SasError::InvalidFormat("No columns found in file".into()));
        }

        // Clone column info to avoid borrow checker issues
        let col_info: Vec<(String, ColumnType, usize, usize)> = self.columns
            .iter()
            .map(|c| (c.name.clone(), c.col_type, c.offset, c.length))
            .collect();

        // Initialize column data storage
        let mut numeric_cols: Vec<Vec<Option<f64>>> = Vec::new();
        let mut string_cols: Vec<Vec<Option<String>>> = Vec::new();
        let mut col_is_numeric: Vec<bool> = Vec::new();

        for (_, col_type, _, _) in &col_info {
            col_is_numeric.push(*col_type == ColumnType::Numeric);
            if *col_type == ColumnType::Numeric {
                numeric_cols.push(Vec::with_capacity(self.row_count));
                string_cols.push(Vec::new()); // Placeholder
            } else {
                numeric_cols.push(Vec::new()); // Placeholder
                string_cols.push(Vec::with_capacity(self.row_count));
            }
        }

        // Read data pages
        let page_header_size = if self.is_64bit { 40 } else { 24 };

        for page_idx in 0..self.page_count {
            let page_offset = self.header_length + page_idx * self.page_size;
            self.reader.seek(SeekFrom::Start(page_offset as u64))?;

            // Read page type
            let page_type_offset = if self.is_64bit { 32 } else { 16 };
            self.reader.seek(SeekFrom::Start((page_offset + page_type_offset) as u64))?;
            let page_type = Self::read_i16(&mut self.reader, self.byte_order)?;

            // Data pages (256) or mixed pages (512)
            if page_type != 256 && page_type != 512 {
                continue;
            }

            // Calculate data start offset
            let data_start = if page_type == 256 {
                page_offset + page_header_size
            } else {
                // For mixed pages, skip subheader pointers
                let sh_count_offset = if self.is_64bit { 34 } else { 18 };
                self.reader.seek(SeekFrom::Start((page_offset + sh_count_offset) as u64))?;
                let sh_count = Self::read_i16(&mut self.reader, self.byte_order)? as usize;
                let subheader_ptr_size = if self.is_64bit { 24 } else { 12 };
                page_offset + page_header_size + sh_count * subheader_ptr_size
            };

            // Read block count for data pages
            let block_count_offset = if self.is_64bit { 36 } else { 20 };
            self.reader.seek(SeekFrom::Start((page_offset + block_count_offset) as u64))?;
            let block_count = Self::read_i16(&mut self.reader, self.byte_order)? as usize;

            // Read rows
            for row_idx in 0..block_count {
                if numeric_cols.iter().filter(|c| !c.is_empty()).map(|c| c.len()).next().unwrap_or(0) >= self.row_count {
                    break;
                }

                let row_offset = data_start + row_idx * self.row_length;

                for (col_idx, (_, col_type, col_offset, col_length)) in col_info.iter().enumerate() {
                    let value_offset = row_offset + col_offset;
                    self.reader.seek(SeekFrom::Start(value_offset as u64))?;

                    if *col_type == ColumnType::Numeric {
                        let value = self.read_numeric_value(*col_length)?;
                        numeric_cols[col_idx].push(value);
                    } else {
                        let value = self.read_string_value(*col_length)?;
                        string_cols[col_idx].push(value);
                    }
                }
            }
        }

        // Build DataFrame columns
        let mut df_columns: Vec<Column> = Vec::with_capacity(col_info.len());

        for (col_idx, (col_name, col_type, _, _)) in col_info.iter().enumerate() {
            let column = if *col_type == ColumnType::Numeric {
                Column::new(col_name.as_str().into(), &numeric_cols[col_idx])
            } else {
                Column::new(col_name.as_str().into(), &string_cols[col_idx])
            };
            df_columns.push(column);
        }

        DataFrame::new(df_columns).map_err(SasError::PolarsError)
    }

    /// Read a numeric value.
    fn read_numeric_value(&mut self, length: usize) -> Result<Option<f64>, SasError> {
        if length == 8 {
            let value = Self::read_f64(&mut self.reader, self.byte_order)?;
            if value.is_nan() || value > 8.988e307 {
                Ok(None)
            } else {
                Ok(Some(value))
            }
        } else if length == 4 {
            let value = Self::read_f32(&mut self.reader, self.byte_order)?;
            if value.is_nan() {
                Ok(None)
            } else {
                Ok(Some(value as f64))
            }
        } else {
            // Read truncated double
            let mut buf = vec![0u8; 8];
            self.reader.read_exact(&mut buf[8 - length..])?;
            let value = match self.byte_order {
                ByteOrder::LittleEndian => f64::from_le_bytes(buf.try_into().unwrap()),
                ByteOrder::BigEndian => f64::from_be_bytes(buf.try_into().unwrap()),
            };
            if value.is_nan() {
                Ok(None)
            } else {
                Ok(Some(value))
            }
        }
    }

    /// Read a string value.
    fn read_string_value(&mut self, length: usize) -> Result<Option<String>, SasError> {
        let mut buf = vec![0u8; length];
        self.reader.read_exact(&mut buf)?;

        // Trim trailing spaces and nulls
        let end = buf.iter().rposition(|&b| b != 0 && b != 0x20).map(|i| i + 1).unwrap_or(0);

        if end == 0 {
            Ok(None)
        } else {
            String::from_utf8(buf[..end].to_vec())
                .or_else(|_| {
                    // Try Windows-1252 encoding
                    Ok(buf[..end].iter().map(|&b| b as char).collect())
                })
                .map(Some)
        }
    }

    fn read_i16(reader: &mut R, byte_order: ByteOrder) -> Result<i16, SasError> {
        let mut buf = [0u8; 2];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => i16::from_le_bytes(buf),
            ByteOrder::BigEndian => i16::from_be_bytes(buf),
        })
    }

    fn read_i32(reader: &mut R, byte_order: ByteOrder) -> Result<i32, SasError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => i32::from_le_bytes(buf),
            ByteOrder::BigEndian => i32::from_be_bytes(buf),
        })
    }

    fn read_i64(reader: &mut R, byte_order: ByteOrder) -> Result<i64, SasError> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => i64::from_le_bytes(buf),
            ByteOrder::BigEndian => i64::from_be_bytes(buf),
        })
    }

    fn read_f32(reader: &mut R, byte_order: ByteOrder) -> Result<f32, SasError> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => f32::from_le_bytes(buf),
            ByteOrder::BigEndian => f32::from_be_bytes(buf),
        })
    }

    fn read_f64(reader: &mut R, byte_order: ByteOrder) -> Result<f64, SasError> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(match byte_order {
            ByteOrder::LittleEndian => f64::from_le_bytes(buf),
            ByteOrder::BigEndian => f64::from_be_bytes(buf),
        })
    }
}

/// Load a SAS7BDAT file into a Polars DataFrame.
pub fn load_sas(path: impl AsRef<Path>) -> Result<DataFrame, SasError> {
    let file = std::fs::File::open(path)?;
    let reader = SasReader::new(file)?;
    reader.read_to_dataframe()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        assert_eq!(SAS_MAGIC.len(), 32);
        assert_eq!(SAS_MAGIC[12], 0xc2);
    }
}
