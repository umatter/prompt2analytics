//! SAS7BDAT file reader (pure Rust implementation).
//!
//! Supports reading SAS data files (.sas7bdat) into Polars DataFrames.

use polars::frame::column::Column;
use polars::prelude::*;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

/// Magic number for SAS7BDAT files (32 bytes)
const SAS_MAGIC: [u8; 32] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc2, 0xea, 0x81, 0x60,
    0xb3, 0x14, 0x11, 0xcf, 0xbd, 0x92, 0x08, 0x00, 0x09, 0xc7, 0x31, 0x8c, 0x18, 0x1f, 0x10, 0x11,
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
            return Err(SasError::InvalidFormat(
                "Invalid SAS7BDAT magic number".into(),
            ));
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
            self.reader
                .seek(SeekFrom::Start((page_offset + page_type_offset) as u64))?;
            let page_type = Self::read_i16(&mut self.reader, self.byte_order)?;

            // Only process metadata pages (0) and mixed pages (512)
            if page_type != 0 && page_type != 512 {
                continue;
            }

            // Read subheader count
            let sh_count_offset = if self.is_64bit { 34 } else { 18 };
            self.reader
                .seek(SeekFrom::Start((page_offset + sh_count_offset) as u64))?;
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
                    self.reader
                        .seek(SeekFrom::Start((sh_abs_offset + row_len_offset) as u64))?;
                    self.row_length = if self.is_64bit {
                        Self::read_i64(&mut self.reader, self.byte_order)? as usize
                    } else {
                        Self::read_i32(&mut self.reader, self.byte_order)? as usize
                    };

                    let row_count_offset = if self.is_64bit { 48 } else { 24 };
                    self.reader
                        .seek(SeekFrom::Start((sh_abs_offset + row_count_offset) as u64))?;
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
            let name = col_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("col_{}", i));
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
    fn parse_column_names(
        &mut self,
        sh_offset: usize,
        sh_length: usize,
        names: &mut Vec<String>,
    ) -> Result<(), SasError> {
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
    fn parse_column_attributes(
        &mut self,
        sh_offset: usize,
        sh_length: usize,
        attrs: &mut Vec<(usize, usize, ColumnType)>,
    ) -> Result<(), SasError> {
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
        let col_info: Vec<(String, ColumnType, usize, usize)> = self
            .columns
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
            self.reader
                .seek(SeekFrom::Start((page_offset + page_type_offset) as u64))?;
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
                self.reader
                    .seek(SeekFrom::Start((page_offset + sh_count_offset) as u64))?;
                let sh_count = Self::read_i16(&mut self.reader, self.byte_order)? as usize;
                let subheader_ptr_size = if self.is_64bit { 24 } else { 12 };
                page_offset + page_header_size + sh_count * subheader_ptr_size
            };

            // Read block count for data pages
            let block_count_offset = if self.is_64bit { 36 } else { 20 };
            self.reader
                .seek(SeekFrom::Start((page_offset + block_count_offset) as u64))?;
            let block_count = Self::read_i16(&mut self.reader, self.byte_order)? as usize;

            // Read rows
            for row_idx in 0..block_count {
                if numeric_cols
                    .iter()
                    .filter(|c| !c.is_empty())
                    .map(|c| c.len())
                    .next()
                    .unwrap_or(0)
                    >= self.row_count
                {
                    break;
                }

                let row_offset = data_start + row_idx * self.row_length;

                for (col_idx, (_, col_type, col_offset, col_length)) in col_info.iter().enumerate()
                {
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
            // Read truncated double. Validate the claimed length first so that a
            // malformed file cannot push us into panicking code paths.
            if length == 0 || length > 8 {
                return Err(SasError::InvalidFormat(format!(
                    "truncated double has invalid length {} (expected 1..=8)",
                    length
                )));
            }
            let mut buf = vec![0u8; 8];
            self.reader.read_exact(&mut buf[8 - length..])?;
            let fixed: [u8; 8] = buf.as_slice().try_into().map_err(|_| {
                SasError::InvalidFormat(
                    "internal: truncated-double buffer was not 8 bytes".to_string(),
                )
            })?;
            let value = match self.byte_order {
                ByteOrder::LittleEndian => f64::from_le_bytes(fixed),
                ByteOrder::BigEndian => f64::from_be_bytes(fixed),
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
        let end = buf
            .iter()
            .rposition(|&b| b != 0 && b != 0x20)
            .map(|i| i + 1)
            .unwrap_or(0);

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
    use std::io::Cursor;

    #[test]
    fn test_magic_number() {
        assert_eq!(SAS_MAGIC.len(), 32);
        assert_eq!(SAS_MAGIC[12], 0xc2);
    }

    #[test]
    fn test_sas_invalid_magic() {
        let invalid_data = b"Not a valid SAS file - this is just random text";
        let cursor = Cursor::new(invalid_data.to_vec());
        let result = SasReader::new(cursor);
        assert!(result.is_err());
        match result {
            Err(SasError::InvalidFormat(msg)) => assert!(msg.contains("magic")),
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_sas_truncated_magic() {
        // Less than 32 bytes
        let short_data = vec![0u8; 16];
        let cursor = Cursor::new(short_data);
        let result = SasReader::new(cursor);
        assert!(result.is_err());
    }

    /// Create a minimal valid SAS7BDAT file (32-bit, little-endian).
    /// This is a highly simplified structure for testing basic parsing.
    fn create_minimal_sas_file() -> Vec<u8> {
        let header_length: usize = 8192; // Standard SAS header
        let page_size: usize = 4096; // Single page size
        let page_count: usize = 2; // Metadata + data page

        let mut buf = vec![0u8; header_length + page_size * page_count];

        // === Header (first 8192 bytes) ===
        // Magic number at offset 0
        buf[0..32].copy_from_slice(&SAS_MAGIC);

        // Alignment byte at offset 32 (0x00 = 32-bit)
        buf[32] = 0x00;

        // Align2 byte at offset 35
        buf[35] = 0x00;

        // Endianness at offset 37 (0x01 = little-endian)
        buf[37] = 0x01;

        // Platform (unused, but let's set it)
        buf[39] = b'1'; // Unix

        // Header length at offset 196 (32-bit, no alignment adjustment)
        let header_len_offset = 196;
        buf[header_len_offset..header_len_offset + 4]
            .copy_from_slice(&(header_length as i32).to_le_bytes());

        // Page size at offset 200
        let page_size_offset = 200;
        buf[page_size_offset..page_size_offset + 4]
            .copy_from_slice(&(page_size as i32).to_le_bytes());

        // Page count at offset 204 (32-bit format)
        let page_count_offset = 204;
        buf[page_count_offset..page_count_offset + 4]
            .copy_from_slice(&(page_count as i32).to_le_bytes());

        // === First page (metadata page) at offset header_length ===
        let page0_offset = header_length;

        // Page type at offset 16 (for 32-bit): 0 = metadata page
        buf[page0_offset + 16..page0_offset + 18].copy_from_slice(&0i16.to_le_bytes());

        // Subheader count at offset 18
        buf[page0_offset + 18..page0_offset + 20].copy_from_slice(&4i16.to_le_bytes()); // 4 subheaders

        // Page header size for 32-bit is 24 bytes
        // Subheader pointer size for 32-bit is 12 bytes

        // === Subheader pointers (start at page0_offset + 24) ===
        let sh_ptr_base = page0_offset + 24;

        // Subheader 1: Row size subheader
        // Offset within page (relative to page start)
        let sh1_offset: i32 = 72; // After 4 subheader pointers (24 + 4*12 = 72)
        buf[sh_ptr_base..sh_ptr_base + 4].copy_from_slice(&sh1_offset.to_le_bytes());
        // Length
        buf[sh_ptr_base + 4..sh_ptr_base + 8].copy_from_slice(&32i32.to_le_bytes());

        // Subheader 2: Column size subheader
        let sh2_offset: i32 = 104;
        buf[sh_ptr_base + 12..sh_ptr_base + 16].copy_from_slice(&sh2_offset.to_le_bytes());
        buf[sh_ptr_base + 16..sh_ptr_base + 20].copy_from_slice(&24i32.to_le_bytes());

        // Subheader 3: Column name subheader
        let sh3_offset: i32 = 128;
        buf[sh_ptr_base + 24..sh_ptr_base + 28].copy_from_slice(&sh3_offset.to_le_bytes());
        buf[sh_ptr_base + 28..sh_ptr_base + 32].copy_from_slice(&40i32.to_le_bytes());

        // Subheader 4: Column attributes subheader
        let sh4_offset: i32 = 168;
        buf[sh_ptr_base + 36..sh_ptr_base + 40].copy_from_slice(&sh4_offset.to_le_bytes());
        buf[sh_ptr_base + 40..sh_ptr_base + 44].copy_from_slice(&36i32.to_le_bytes());

        // === Row size subheader (F7F7F7F7) ===
        let row_sh_base = page0_offset + sh1_offset as usize;
        buf[row_sh_base..row_sh_base + 4].copy_from_slice(&ROW_SIZE_SIGNATURE);
        // Row length at offset 20 (32-bit)
        let row_length: i32 = 16; // 2 doubles = 16 bytes per row
        buf[row_sh_base + 20..row_sh_base + 24].copy_from_slice(&row_length.to_le_bytes());
        // Row count at offset 24
        let row_count: i32 = 3; // 3 rows
        buf[row_sh_base + 24..row_sh_base + 28].copy_from_slice(&row_count.to_le_bytes());

        // === Column size subheader (F6F6F6F6) ===
        let col_sh_base = page0_offset + sh2_offset as usize;
        buf[col_sh_base..col_sh_base + 4].copy_from_slice(&COL_SIZE_SIGNATURE);
        // Column count at offset 8
        let col_count: i32 = 2;
        buf[col_sh_base + 8..col_sh_base + 12].copy_from_slice(&col_count.to_le_bytes());

        // === Column name subheader (FFFFFFFF) ===
        let name_sh_base = page0_offset + sh3_offset as usize;
        buf[name_sh_base..name_sh_base + 4].copy_from_slice(&COL_NAME_SIGNATURE);
        // Base offset for names is 8 + int_size*2 = 8 + 4*2 = 16
        // Entry size is 8 bytes
        // We have 2 columns

        // === Column attributes subheader (FCFFFFFF) ===
        let attr_sh_base = page0_offset + sh4_offset as usize;
        buf[attr_sh_base..attr_sh_base + 4].copy_from_slice(&COL_ATTR_SIGNATURE);
        // Base offset is 12 for 32-bit
        // Each attribute entry is 12 bytes for 32-bit

        // Column 1: offset=0, width=8, type=1 (numeric)
        let attr1_base = attr_sh_base + 12;
        buf[attr1_base..attr1_base + 4].copy_from_slice(&0i32.to_le_bytes()); // offset
        buf[attr1_base + 4..attr1_base + 8].copy_from_slice(&8i32.to_le_bytes()); // width
        buf[attr1_base + 8] = 1; // type = numeric

        // Column 2: offset=8, width=8, type=1 (numeric)
        let attr2_base = attr_sh_base + 24;
        buf[attr2_base..attr2_base + 4].copy_from_slice(&8i32.to_le_bytes()); // offset
        buf[attr2_base + 4..attr2_base + 8].copy_from_slice(&8i32.to_le_bytes()); // width
        buf[attr2_base + 8] = 1; // type = numeric

        // === Second page (data page) at offset header_length + page_size ===
        let page1_offset = header_length + page_size;

        // Page type at offset 16: 256 = data page
        buf[page1_offset + 16..page1_offset + 18].copy_from_slice(&256i16.to_le_bytes());

        // Block count (number of rows in this page) at offset 20
        buf[page1_offset + 20..page1_offset + 22].copy_from_slice(&3i16.to_le_bytes());

        // Data starts at offset 24 (page_header_size for 32-bit)
        let data_base = page1_offset + 24;

        // Row 1: 1.0, 2.0
        buf[data_base..data_base + 8].copy_from_slice(&1.0f64.to_le_bytes());
        buf[data_base + 8..data_base + 16].copy_from_slice(&2.0f64.to_le_bytes());

        // Row 2: 3.0, 4.0
        buf[data_base + 16..data_base + 24].copy_from_slice(&3.0f64.to_le_bytes());
        buf[data_base + 24..data_base + 32].copy_from_slice(&4.0f64.to_le_bytes());

        // Row 3: 5.0, 6.0
        buf[data_base + 32..data_base + 40].copy_from_slice(&5.0f64.to_le_bytes());
        buf[data_base + 40..data_base + 48].copy_from_slice(&6.0f64.to_le_bytes());

        buf
    }

    #[test]
    fn test_read_synthetic_sas_header() {
        let sas_bytes = create_minimal_sas_file();
        let cursor = Cursor::new(sas_bytes);
        let reader = SasReader::new(cursor).expect("Failed to create SAS reader");

        // Verify header parsing
        assert!(!reader.is_64bit);
        assert_eq!(reader.byte_order, ByteOrder::LittleEndian);
        assert_eq!(reader.header_length, 8192);
        assert_eq!(reader.page_size, 4096);
        assert_eq!(reader.page_count, 2);
        assert_eq!(reader.row_length, 16);
        assert_eq!(reader.row_count, 3);
        assert_eq!(reader.columns.len(), 2);
    }

    #[test]
    fn test_read_synthetic_sas_data() {
        let sas_bytes = create_minimal_sas_file();
        let cursor = Cursor::new(sas_bytes);
        let reader = SasReader::new(cursor).expect("Failed to create SAS reader");
        let df = reader
            .read_to_dataframe()
            .expect("Failed to read dataframe");

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 2);

        // Get columns by index since names are auto-generated
        let columns: Vec<_> = df.get_columns().iter().collect();
        assert_eq!(columns.len(), 2);

        // First column should have values 1.0, 3.0, 5.0
        let col1 = columns[0].f64().unwrap();
        assert_eq!(col1.get(0), Some(1.0));
        assert_eq!(col1.get(1), Some(3.0));
        assert_eq!(col1.get(2), Some(5.0));

        // Second column should have values 2.0, 4.0, 6.0
        let col2 = columns[1].f64().unwrap();
        assert_eq!(col2.get(0), Some(2.0));
        assert_eq!(col2.get(1), Some(4.0));
        assert_eq!(col2.get(2), Some(6.0));
    }

    #[test]
    fn test_sas_endianness_detection_little() {
        // Test little-endian (default)
        let sas_bytes = create_minimal_sas_file();
        let cursor = Cursor::new(sas_bytes);
        let reader = SasReader::new(cursor).expect("Failed to create reader");
        assert_eq!(reader.byte_order, ByteOrder::LittleEndian);
    }

    #[test]
    fn test_sas_column_types() {
        // Verify that column type detection works
        let sas_bytes = create_minimal_sas_file();
        let cursor = Cursor::new(sas_bytes);
        let reader = SasReader::new(cursor).expect("Failed to create reader");

        // Verify we have 2 numeric columns
        assert_eq!(reader.columns.len(), 2);
        assert_eq!(reader.columns[0].col_type, ColumnType::Numeric);
        assert_eq!(reader.columns[1].col_type, ColumnType::Numeric);
    }
}
