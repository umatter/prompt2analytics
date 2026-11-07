//! SAS7BDAT file reader (pure Rust implementation).
//!
//! Reads uncompressed files, 32- and 64-bit, little- and big-endian. The layout follows ReadStat
//! (`readstat_sas.c`, `readstat_sas7bdat_read.c`) and pandas (`pandas/io/sas/sas7bdat.py`). Every
//! case is pinned to `haven::read_sas` on the fixtures in `validation/datasets/files/`
//! (`validation/data/readers.md`).
//!
//! | Field | Where |
//! |---|---|
//! | 64-bit integers | byte 32 is 0x33: offsets, lengths and counts are 8 bytes, else 4 |
//! | header pad | byte 35 is 0x33: 4 bytes follow byte 164 (`pad`) |
//! | byte order | byte 37: 0x01 little-endian, 0x00 big-endian |
//! | character encoding | byte 70: 20 UTF-8, 28 US-ASCII, 29 ISO-8859-1, 0, 62 and 204 Windows-1252 |
//! | header length, page size | u32 at 196 + pad and 200 + pad |
//! | page count | u32 or u64 at 204 + pad |
//! | page header | type, block count, subheader count: u16 at B, B + 2, B + 4 (B = 16 or 32) |
//! | subheader pointer | offset, length (int), compression (u8), type (u8), from B + 8 |
//!
//! Page type 0 is a metadata page, 256 a data page and 512 a mix page; other pages are skipped. A
//! data page holds `block count` rows after its header. A mix page holds
//! `block count - subheader count` rows after its pointer table.
//!
//! Text cells are trimmed of trailing blanks and NULs; an empty cell is `""`, as haven reads it. Every NaN numeric (SAS `.`, `.A`-`.Z`, `._`) is null and every other value is
//! kept (1e308 is a valid SAS value). Numerics of 3 to 7 bytes are doubles truncated to their
//! high-order bytes.

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

/// Bytes of the file header the reader parses (the fields end before byte 288 with the pad).
const HEADER_PREFIX: usize = 288;

/// Subheader signatures: the first 32-bit word in the file's byte order (the second word of a
/// big-endian 64-bit signature whose first word is 0 or all ones).
const SIG_ROW_SIZE: u32 = 0xF7F7_F7F7;
const SIG_COL_SIZE: u32 = 0xF6F6_F6F6;
const SIG_COL_TEXT: u32 = 0xFFFF_FFFD;
const SIG_COL_NAME: u32 = 0xFFFF_FFFF;
const SIG_COL_ATTR: u32 = 0xFFFF_FFFC;

/// Subheader pointer compression codes.
const COMPRESSION_NONE: u8 = 0;
const COMPRESSION_TRUNCATED: u8 = 1;

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

/// A signed size or offset field read from the file, as `usize`: negative
/// values are a format error instead of wrapping to a huge `usize`.
fn field_usize(raw: i64, name: &str) -> Result<usize, SasError> {
    if raw < 0 {
        return Err(SasError::InvalidFormat(format!(
            "{name} = {raw} is negative; the file is corrupt"
        )));
    }
    usize::try_from(raw).map_err(|_| {
        SasError::InvalidFormat(format!("{name} = {raw} is too large; the file is corrupt"))
    })
}

fn offset_overflow() -> SasError {
    SasError::InvalidFormat("offset arithmetic overflowed; the file is corrupt".to_string())
}

/// Checked `a + b` for offsets computed from file fields.
fn add(a: usize, b: usize) -> Result<usize, SasError> {
    a.checked_add(b).ok_or_else(offset_overflow)
}

/// Checked `a * b` for offsets computed from file fields.
fn mul(a: usize, b: usize) -> Result<usize, SasError> {
    a.checked_mul(b).ok_or_else(offset_overflow)
}

/// Map a core resource error (allocation size, cancellation) to a parse
/// error: a large valid file is not a corrupt one.
fn resource_err(e: crate::errors::EconError) -> SasError {
    SasError::ParseError(e.to_string())
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

/// What a page holds.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PageKind {
    Meta,
    Data,
    Mix,
    Skip,
}

/// Column metadata
#[derive(Debug, Clone)]
struct ColumnInfo {
    name: String,
    col_type: ColumnType,
    offset: usize,
    length: usize,
}

/// The header fields the page helpers need, as one `Copy` value.
#[derive(Debug, Clone, Copy)]
struct Layout {
    is_64bit: bool,
    byte_order: ByteOrder,
    encoding: u8,
    page_size: usize,
    page_count: usize,
}

impl Layout {
    fn int_len(&self) -> usize {
        if self.is_64bit { 8 } else { 4 }
    }
    fn page_type_offset(&self) -> usize {
        if self.is_64bit { 32 } else { 16 }
    }
    fn page_header_size(&self) -> usize {
        self.page_type_offset() + 8
    }
    fn pointer_size(&self) -> usize {
        3 * self.int_len()
    }
}

/// One subheader pointer of a page.
#[derive(Debug, Clone, Copy)]
struct Pointer {
    offset: usize,
    length: usize,
    compression: u8,
    kind: u8,
}

/// Bounds-checked unsigned read of `width` (1, 2, 4 or 8) bytes at `at` of `buf`.
fn read_uint(buf: &[u8], at: usize, width: usize, order: ByteOrder) -> Result<u64, SasError> {
    let end = at.checked_add(width).ok_or_else(offset_overflow)?;
    let bytes = buf.get(at..end).ok_or_else(|| {
        SasError::InvalidFormat(format!(
            "a {width}-byte field at offset {at} lies outside its {}-byte block",
            buf.len()
        ))
    })?;
    Ok(match order {
        ByteOrder::LittleEndian => bytes
            .iter()
            .rev()
            .fold(0u64, |v, &b| (v << 8) | u64::from(b)),
        ByteOrder::BigEndian => bytes.iter().fold(0u64, |v, &b| (v << 8) | u64::from(b)),
    })
}

/// Signed read (sign-extended from `width` bytes): fields the T0 checks reject when negative.
fn read_int(buf: &[u8], at: usize, width: usize, order: ByteOrder) -> Result<i64, SasError> {
    let v = read_uint(buf, at, width, order)?;
    let shift = 64 - 8 * (width as u32);
    Ok(((v << shift) as i64) >> shift)
}

fn read_u16(buf: &[u8], at: usize, order: ByteOrder) -> Result<usize, SasError> {
    Ok(read_uint(buf, at, 2, order)? as usize)
}

/// Windows-1252 code points of bytes 0x80..=0x9F; `None` for the five bytes the code page leaves
/// unassigned (0x81, 0x8D, 0x8F, 0x90, 0x9D). Bytes 0xA0..=0xFF are ISO-8859-1.
const WINDOWS_1252_80_9F: [Option<char>; 32] = [
    Some('\u{20AC}'),
    None,
    Some('\u{201A}'),
    Some('\u{192}'),
    Some('\u{201E}'),
    Some('\u{2026}'),
    Some('\u{2020}'),
    Some('\u{2021}'),
    Some('\u{2C6}'),
    Some('\u{2030}'),
    Some('\u{160}'),
    Some('\u{2039}'),
    Some('\u{152}'),
    None,
    Some('\u{17D}'),
    None,
    None,
    Some('\u{2018}'),
    Some('\u{2019}'),
    Some('\u{201C}'),
    Some('\u{201D}'),
    Some('\u{2022}'),
    Some('\u{2013}'),
    Some('\u{2014}'),
    Some('\u{2DC}'),
    Some('\u{2122}'),
    Some('\u{161}'),
    Some('\u{203A}'),
    Some('\u{153}'),
    None,
    Some('\u{17E}'),
    Some('\u{178}'),
];

/// Strict Windows-1252: `Err(byte)` at the first unassigned byte (as iconv, and so ReadStat).
pub(crate) fn decode_windows_1252(bytes: &[u8]) -> Result<String, u8> {
    bytes
        .iter()
        .map(|&b| match b {
            0x80..=0x9F => WINDOWS_1252_80_9F[usize::from(b - 0x80)].ok_or(b),
            _ => Ok(char::from(b)),
        })
        .collect()
}

/// Windows-1252 that maps the five unassigned bytes to the C1 control of the same value (as the
/// WHATWG encoding standard): for text whose encoding is a guess (Stata release 117) and for
/// showing raw bytes in error messages.
pub(crate) fn decode_windows_1252_lenient(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| match b {
            0x80..=0x9F => WINDOWS_1252_80_9F[usize::from(b - 0x80)].unwrap_or(char::from(b)),
            _ => char::from(b),
        })
        .collect()
}

/// UTF-8 when the bytes are valid UTF-8, Windows-1252 otherwise (Stata release 117 text: Stata 13
/// writes the platform encoding, haven and ReadStat write UTF-8).
pub(crate) fn decode_utf8_or_windows_1252(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => decode_windows_1252_lenient(bytes),
    }
}

/// Decode SAS text by the header's encoding byte, as ReadStat does (verified byte by byte against
/// haven::read_sas, validation/data/readers.md): 20 UTF-8; 28 US-ASCII; 29 ISO-8859-1; 0, 62 and
/// 204 Windows-1252. ASCII text reads the same under every code; other non-ASCII text is refused
/// with the code named.
fn decode_sas_text(bytes: &[u8], encoding: u8) -> Result<String, String> {
    if bytes.is_ascii() {
        return Ok(String::from_utf8_lossy(bytes).into_owned());
    }
    match encoding {
        20 => std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| "the file is declared UTF-8 but this text is not valid UTF-8".to_string()),
        29 => Ok(bytes.iter().map(|&b| char::from(b)).collect()),
        0 | 62 | 204 => decode_windows_1252(bytes).map_err(|b| {
            format!(
                "byte 0x{b:02X} is not a Windows-1252 character (the file declares Windows-1252)"
            )
        }),
        28 => Err("the file is declared US-ASCII but this text is not ASCII".to_string()),
        other => Err(format!(
            "SAS character encoding code {other} is not supported for non-ASCII text (UTF-8 = 20, \
             US-ASCII = 28, ISO-8859-1 = 29 and Windows-1252 = 0, 62, 204 are)"
        )),
    }
}

/// Trim trailing blanks and NULs (SAS pads text with either).
fn trim_text(bytes: &[u8]) -> &[u8] {
    let end = bytes
        .iter()
        .rposition(|&b| b != 0 && b != b' ')
        .map_or(0, |i| i + 1);
    &bytes[..end]
}

/// A SAS numeric of `cell.len()` (1..=8) bytes: the high-order bytes of an IEEE double, at the end
/// of the 8 bytes in little-endian files and at the start in big-endian files (a LENGTH 4 numeric
/// is a truncated double, not a float). Every NaN (SAS `.`, `.A`-`.Z`, `._`) is missing.
fn decode_numeric(cell: &[u8], order: ByteOrder) -> Option<f64> {
    let w = cell.len();
    if w == 0 || w > 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    let v = match order {
        ByteOrder::LittleEndian => {
            buf[8 - w..].copy_from_slice(cell);
            f64::from_le_bytes(buf)
        }
        ByteOrder::BigEndian => {
            buf[..w].copy_from_slice(cell);
            f64::from_be_bytes(buf)
        }
    };
    (!v.is_nan()).then_some(v)
}

/// The kind of a page from its type word.
fn page_kind(page_type: u16) -> PageKind {
    match page_type {
        0 => PageKind::Meta,
        256 => PageKind::Data,
        512 => PageKind::Mix,
        _ => PageKind::Skip,
    }
}

/// Column buffers filled row by row.
struct RowSink {
    numeric: Vec<Vec<Option<f64>>>,
    text: Vec<Vec<Option<String>>>,
    rows: usize,
}

/// SAS7BDAT file reader
pub struct SasReader<R: Read + Seek> {
    reader: R,
    byte_order: ByteOrder,
    is_64bit: bool,
    /// Header encoding byte (70).
    encoding: u8,
    header_length: usize,
    page_size: usize,
    page_count: usize,
    row_length: usize,
    row_count: usize,
    columns: Vec<ColumnInfo>,
}

impl<R: Read + Seek> SasReader<R> {
    /// Create a new SAS reader: parses the header and every metadata subheader.
    pub fn new(mut reader: R) -> Result<Self, SasError> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(0))?;
        let mut head = [0u8; HEADER_PREFIX];
        let mut got = 0;
        while got < head.len() {
            let n = reader.read(&mut head[got..])?;
            if n == 0 {
                break;
            }
            got += n;
        }
        if got < SAS_MAGIC.len() || head[..SAS_MAGIC.len()] != SAS_MAGIC {
            return Err(SasError::InvalidFormat(
                "Invalid SAS7BDAT magic number".into(),
            ));
        }
        if got < head.len() {
            return Err(SasError::InvalidFormat(format!(
                "the file ends inside its header ({got} bytes)"
            )));
        }
        let is_64bit = head[32] == 0x33;
        let pad = if head[35] == 0x33 { 4 } else { 0 };
        let byte_order = match head[37] {
            0x01 => ByteOrder::LittleEndian,
            0x00 => ByteOrder::BigEndian,
            b => {
                return Err(SasError::InvalidFormat(format!(
                    "byte 37 (byte order) is 0x{b:02x}, not 0x00 or 0x01"
                )));
            }
        };
        let encoding = head[70];

        // Header fields, validated against the file size (T0, DATA-49).
        let check_offset = |raw: u64, name: &str| -> Result<usize, SasError> {
            if raw > file_len {
                return Err(SasError::InvalidFormat(format!(
                    "{name} = {raw} exceeds the file size ({file_len} bytes); \
                     the file is corrupt or truncated"
                )));
            }
            usize::try_from(raw).map_err(|_| offset_overflow())
        };
        let header_length =
            check_offset(read_uint(&head, 196 + pad, 4, byte_order)?, "header_length")?;
        let page_size = check_offset(read_uint(&head, 200 + pad, 4, byte_order)?, "page_size")?;
        if page_size == 0 {
            return Err(SasError::InvalidFormat("page_size is zero".to_string()));
        }
        let int_len = if is_64bit { 8 } else { 4 };
        let page_count_raw = read_int(&head, 204 + pad, int_len, byte_order)?;
        if page_count_raw < 0 {
            return Err(SasError::InvalidFormat(format!(
                "page_count = {page_count_raw} is negative; the file is corrupt"
            )));
        }
        let page_count = usize::try_from(page_count_raw).map_err(|_| offset_overflow())?;
        // A page_count larger than the file can hold indicates corruption and
        // would drive out-of-range page reads.
        let max_pages = file_len.saturating_sub(header_length as u64) / page_size as u64;
        if page_count as u64 > max_pages {
            return Err(SasError::InvalidFormat(format!(
                "declared page_count {page_count} exceeds file capacity ({max_pages} pages of \
                 {page_size} bytes after a {header_length}-byte header in a {file_len}-byte file)"
            )));
        }

        let mut sas_reader = SasReader {
            reader,
            byte_order,
            is_64bit,
            encoding,
            header_length,
            page_size,
            page_count,
            row_length: 0,
            row_count: 0,
            columns: Vec::new(),
        };
        sas_reader.parse_metadata()?;
        Ok(sas_reader)
    }

    /// The header fields as one `Copy` value for the page helpers.
    fn layout(&self) -> Layout {
        Layout {
            is_64bit: self.is_64bit,
            byte_order: self.byte_order,
            encoding: self.encoding,
            page_size: self.page_size,
            page_count: self.page_count,
        }
    }

    /// Read page `idx` into `page` (`page_size` bytes).
    fn read_page(&mut self, idx: usize, page: &mut [u8]) -> Result<(), SasError> {
        let at = add(self.header_length, mul(idx, self.page_size)?)?;
        self.reader.seek(SeekFrom::Start(at as u64))?;
        self.reader.read_exact(page)?;
        Ok(())
    }

    /// The page's kind, block count and subheader pointers (none on data and skipped pages).
    fn page_header(
        &self,
        page: &[u8],
        idx: usize,
    ) -> Result<(PageKind, usize, Vec<Pointer>), SasError> {
        let l = self.layout();
        let t = l.page_type_offset();
        let kind = page_kind(read_uint(page, t, 2, l.byte_order)? as u16);
        let block_count = read_u16(page, t + 2, l.byte_order)?;
        let subheader_count = read_u16(page, t + 4, l.byte_order)?;
        let mut pointers = Vec::new();
        if matches!(kind, PageKind::Meta | PageKind::Mix) {
            // The pointer table must lie inside the page.
            if add(
                l.page_header_size(),
                mul(subheader_count, l.pointer_size())?,
            )? > l.page_size
            {
                return Err(SasError::InvalidFormat(format!(
                    "subheader count {subheader_count} on page {idx} does not fit in the page"
                )));
            }
            let il = l.int_len();
            for k in 0..subheader_count {
                let q = l.page_header_size() + k * l.pointer_size();
                let offset = field_usize(read_int(page, q, il, l.byte_order)?, "subheader offset")?;
                let length = field_usize(
                    read_int(page, q + il, il, l.byte_order)?,
                    "subheader length",
                )?;
                let compression = page[q + 2 * il];
                let kind = page[q + 2 * il + 1];
                // A subheader must lie inside its page (a truncated one is never read).
                if length > 0
                    && compression != COMPRESSION_TRUNCATED
                    && add(offset, length)? > l.page_size
                {
                    return Err(SasError::InvalidFormat(format!(
                        "subheader {k} on page {idx} extends past the page"
                    )));
                }
                pointers.push(Pointer {
                    offset,
                    length,
                    compression,
                    kind,
                });
            }
        }
        Ok((kind, block_count, pointers))
    }

    /// The subheader signature at the start of `sh`.
    fn signature(&self, sh: &[u8]) -> Result<u32, SasError> {
        let l = self.layout();
        let first = read_uint(sh, 0, 4, l.byte_order)? as u32;
        if l.is_64bit
            && l.byte_order == ByteOrder::BigEndian
            && (first == 0xFFFF_FFFF || first == 0)
        {
            return Ok(read_uint(sh, 4, 4, l.byte_order)? as u32);
        }
        Ok(first)
    }

    /// Pass 1: row size, column count, column text, names and attributes from every metadata
    /// and mix page; then the names are resolved and the layout validated.
    fn parse_metadata(&mut self) -> Result<(), SasError> {
        let l = self.layout();
        let il = l.int_len();
        let mut page =
            crate::resource::try_vec(l.page_size, 0u8, "SAS page").map_err(resource_err)?;
        let (mut row_length, mut row_count, mut column_count) = (None, None, None);
        let mut texts: Vec<Vec<u8>> = Vec::new();
        let mut names: Vec<(usize, usize, usize)> = Vec::new(); // (text block, offset, length)
        let mut attrs: Vec<(usize, usize, ColumnType)> = Vec::new(); // (offset, width, type)

        for idx in 0..l.page_count {
            crate::resource::checkpoint().map_err(resource_err)?;
            self.read_page(idx, &mut page)?;
            let (kind, _, pointers) = self.page_header(&page, idx)?;
            if !matches!(kind, PageKind::Meta | PageKind::Mix) {
                continue;
            }
            for p in pointers {
                // Truncated subheaders carry no metadata.
                if p.length == 0 || p.compression != COMPRESSION_NONE {
                    continue;
                }
                let sh = &page[p.offset..p.offset + p.length];
                let sig = self.signature(sh)?;
                let count_of = |entry: usize| -> Result<usize, SasError> {
                    p.length
                        .checked_sub(2 * il + 12)
                        .map(|r| r / entry)
                        .ok_or_else(|| {
                            SasError::InvalidFormat(format!(
                                "subheader 0x{sig:08x} on page {idx} is only {} bytes long",
                                p.length
                            ))
                        })
                };
                match sig {
                    SIG_ROW_SIZE => {
                        row_length = Some(field_usize(
                            read_int(sh, 5 * il, il, l.byte_order)?,
                            "row_length",
                        )?);
                        row_count = Some(field_usize(
                            read_int(sh, 6 * il, il, l.byte_order)?,
                            "row_count",
                        )?);
                    }
                    SIG_COL_SIZE => {
                        column_count = Some(field_usize(
                            read_int(sh, il, il, l.byte_order)?,
                            "column count",
                        )?);
                    }
                    SIG_COL_TEXT => {
                        let blob = sh.get(il..).unwrap_or(&[]).to_vec();
                        texts.push(blob);
                    }
                    SIG_COL_NAME => {
                        for i in 0..count_of(8)? {
                            let q = il + 8 + 8 * i;
                            names.push((
                                read_u16(sh, q, l.byte_order)?,
                                read_u16(sh, q + 2, l.byte_order)?,
                                read_u16(sh, q + 4, l.byte_order)?,
                            ));
                        }
                    }
                    SIG_COL_ATTR => {
                        for i in 0..count_of(il + 8)? {
                            let q = il + 8 + i * (il + 8);
                            let offset =
                                field_usize(read_int(sh, q, il, l.byte_order)?, "column offset")?;
                            let width = field_usize(
                                read_int(sh, q + il, 4, l.byte_order)?,
                                "column width",
                            )?;
                            let code = read_uint(sh, q + il + 6, 1, l.byte_order)?;
                            let col_type = match code {
                                1 => ColumnType::Numeric,
                                2 => ColumnType::Character,
                                other => {
                                    return Err(SasError::InvalidFormat(format!(
                                        "column {} has the unknown type code {other} \
                                         (1 = numeric, 2 = character)",
                                        attrs.len()
                                    )));
                                }
                            };
                            attrs.push((offset, width, col_type));
                        }
                    }
                    _ => {}
                }
            }
        }

        if attrs.is_empty() {
            return Err(SasError::InvalidFormat("No columns found in file".into()));
        }
        self.row_length = row_length
            .ok_or_else(|| SasError::InvalidFormat("the file has no row size subheader".into()))?;
        self.row_count = row_count.unwrap_or(0);
        if names.len() != attrs.len() || column_count.is_some_and(|c| c != attrs.len()) {
            return Err(SasError::InvalidFormat(format!(
                "the file describes {} columns but names {} (column count {})",
                attrs.len(),
                names.len(),
                column_count.map_or_else(|| "absent".to_string(), |c| c.to_string())
            )));
        }
        for (j, ((t, off, len), (offset, length, col_type))) in names.iter().zip(&attrs).enumerate()
        {
            let raw = texts
                .get(*t)
                .and_then(|b| b.get(*off..off + len))
                .ok_or_else(|| {
                    SasError::InvalidFormat(format!(
                        "the name of column {j} points outside the column text"
                    ))
                })?;
            let name = decode_sas_text(trim_text(raw), l.encoding)
                .map_err(|e| SasError::ParseError(format!("name of column {j}: {e}")))?;
            self.columns.push(ColumnInfo {
                name,
                col_type: *col_type,
                offset: *offset,
                length: *length,
            });
        }
        self.validate_layout()
    }

    /// Check the row and column layout against the page geometry: the row length is between 1
    /// and the page size, the row count fits the pages,
    /// every column has a nonzero width and lies inside the row, numeric widths are 1 to 8 bytes,
    /// and the column widths add up to at most the row length. So the bytes read per row are
    /// bounded by the row length, and the cells read by the file size (T0, DATA-49). A valid file
    /// satisfies all of these. Columns are not checked for disjointness: two overlapping columns
    /// read the same bytes of the row.
    fn validate_layout(&self) -> Result<(), SasError> {
        let l = self.layout();
        let row_length = self.row_length;
        if row_length == 0 {
            return Err(SasError::InvalidFormat("row_length is zero".to_string()));
        }
        if row_length > l.page_size {
            return Err(SasError::InvalidFormat(format!(
                "row_length {row_length} exceeds the page size of {} bytes",
                l.page_size
            )));
        }
        let rows_per_page = l.page_size / row_length;
        let max_rows = l.page_count.saturating_mul(rows_per_page);
        if self.row_count > max_rows {
            return Err(SasError::InvalidFormat(format!(
                "row_count {} exceeds what {} pages of {} bytes can hold",
                self.row_count, l.page_count, l.page_size
            )));
        }

        let mut total_width = 0usize;
        for (i, column) in self.columns.iter().enumerate() {
            let (offset, width) = (column.offset, column.length);
            if width == 0 {
                return Err(SasError::InvalidFormat(format!(
                    "column {i} (offset {offset}) has width 0"
                )));
            }
            if add(offset, width)? > row_length {
                return Err(SasError::InvalidFormat(format!(
                    "column {i} (offset {offset}, width {width}) extends past the row_length \
                     of {row_length}"
                )));
            }
            if column.col_type == ColumnType::Numeric && width > 8 {
                return Err(SasError::InvalidFormat(format!(
                    "numeric column {} has width {width}; SAS numerics are 1 to 8 bytes",
                    column.name
                )));
            }
            total_width = add(total_width, width)?;
        }
        if total_width > row_length {
            return Err(SasError::InvalidFormat(format!(
                "column widths add up to {total_width} bytes, more than the row_length of \
                 {row_length}"
            )));
        }
        Ok(())
    }

    /// Decode one row into the column buffers.
    fn push_row(&self, sink: &mut RowSink, row: &[u8]) -> Result<(), SasError> {
        let order = self.byte_order;
        for (j, c) in self.columns.iter().enumerate() {
            let cell = &row[c.offset..c.offset + c.length];
            match c.col_type {
                ColumnType::Numeric => sink.numeric[j].push(decode_numeric(cell, order)),
                ColumnType::Character => {
                    let value = decode_sas_text(trim_text(cell), self.encoding).map_err(|e| {
                        SasError::ParseError(format!(
                            "column {}, row {}: {e}",
                            c.name,
                            sink.rows + 1
                        ))
                    })?;
                    sink.text[j].push(Some(value));
                }
            }
        }
        sink.rows += 1;
        Ok(())
    }

    /// Read the data into a DataFrame (pass 2: rows in page order).
    pub fn read_to_dataframe(mut self) -> Result<DataFrame, SasError> {
        let l = self.layout();
        // Bound the pre-allocation: `row_count` is parsed from a page subheader
        // and must not be trusted to size buffers directly (a corrupt value
        // would trigger a huge allocation). It grows as rows are actually read.
        let row_capacity = self.row_count.min(1 << 20);
        let cells =
            crate::resource::checked_len(&[self.columns.len(), row_capacity], "SAS columns")
                .map_err(resource_err)?;
        crate::resource::check_alloc::<Option<String>>(cells, "SAS columns")
            .map_err(resource_err)?;
        let mut sink = RowSink {
            numeric: Vec::new(),
            text: Vec::new(),
            rows: 0,
        };
        for c in &self.columns {
            let numeric = c.col_type == ColumnType::Numeric;
            sink.numeric
                .push(Vec::with_capacity(if numeric { row_capacity } else { 0 }));
            sink.text
                .push(Vec::with_capacity(if numeric { 0 } else { row_capacity }));
        }
        let mut page =
            crate::resource::try_vec(l.page_size, 0u8, "SAS page").map_err(resource_err)?;
        let mut row =
            crate::resource::try_vec(self.row_length, 0u8, "SAS row").map_err(resource_err)?;

        for idx in 0..l.page_count {
            if sink.rows >= self.row_count {
                break;
            }
            crate::resource::checkpoint().map_err(resource_err)?;
            self.read_page(idx, &mut page)?;
            let (kind, block_count, pointers) = self.page_header(&page, idx)?;

            // Rows stored after the page header (data pages) or the pointer table (mix pages).
            let (start, n) = match kind {
                PageKind::Data => (l.page_header_size(), block_count),
                PageKind::Mix => {
                    let start = add(l.page_header_size(), mul(pointers.len(), l.pointer_size())?)?;
                    let n = block_count.checked_sub(pointers.len()).ok_or_else(|| {
                        SasError::InvalidFormat(format!(
                            "block count {block_count} on page {idx} is smaller than its {} \
                             subheaders",
                            pointers.len()
                        ))
                    })?;
                    (start, n)
                }
                PageKind::Meta | PageKind::Skip => continue,
            };
            let capacity = l.page_size.saturating_sub(start) / self.row_length;
            if n > capacity {
                return Err(SasError::InvalidFormat(format!(
                    "block count {n} on page {idx} exceeds the {capacity} rows that fit in the page"
                )));
            }
            for r in 0..n {
                if sink.rows >= self.row_count {
                    break;
                }
                let at = start + r * self.row_length;
                row.copy_from_slice(&page[at..at + self.row_length]);
                self.push_row(&mut sink, &row)?;
            }
        }

        let mut df_columns: Vec<Column> = Vec::with_capacity(self.columns.len());
        for (j, c) in self.columns.iter().enumerate() {
            let column = match c.col_type {
                ColumnType::Numeric => Column::new(c.name.as_str().into(), &sink.numeric[j]),
                ColumnType::Character => Column::new(c.name.as_str().into(), &sink.text[j]),
            };
            df_columns.push(column);
        }
        DataFrame::new(df_columns).map_err(SasError::PolarsError)
    }
}

/// Load a SAS7BDAT file into a Polars DataFrame.
pub fn load_sas(path: impl AsRef<Path>) -> Result<DataFrame, SasError> {
    let file = std::fs::File::open(path)?;
    let reader = SasReader::new(file)?;
    reader.read_to_dataframe()
}

/// Test-only comparison of a reader's DataFrame with haven's read of the same committed fixture
/// (case `data_readers_values`, `validation/r/cases/data_readers.R`; the fixtures are in
/// `validation/datasets/files/`, written by `validation/r/fixtures/data_readers.R`). The Stata
/// reader's tests use it too.
#[cfg(test)]
pub(crate) mod fixture_check {
    use p2a_validation::{Case, Check, Tol};
    use polars::prelude::*;

    /// Absolute path of a committed fixture.
    pub(crate) fn fixture_path(name: &str) -> std::path::PathBuf {
        p2a_validation::repo_path(&format!("validation/datasets/files/{name}"))
    }

    /// The bytes of a committed fixture (for tests that patch fields).
    pub(crate) fn fixture_bytes(name: &str) -> Vec<u8> {
        std::fs::read(fixture_path(name)).unwrap_or_else(|e| {
            panic!(
                "fixture {name}: {e}; regenerate with `Rscript validation/r/fixtures/data_readers.R`"
            )
        })
    }

    /// Write `bytes` into `b` at `at`.
    pub(crate) fn put(b: &mut [u8], at: usize, bytes: &[u8]) {
        b[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// `df` equals haven's read of fixture `stem`: the shape, the column names in file order,
    /// every numeric value exactly (null where R has NA) and every string exactly.
    pub(crate) fn assert_matches_haven(case: &Case, stem: &str, df: &DataFrame) {
        let shape = case.vector(&format!("{stem}__shape"));
        assert_eq!(
            (df.height() as f64, df.width() as f64),
            (shape[0], shape[1]),
            "{stem}: rows x columns"
        );
        let columns_key = format!("{stem}__columns");
        let names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            names.as_slice(),
            case.names(&columns_key),
            "{stem}: column names and order"
        );
        let numeric = case.vector(&columns_key);
        let mut check = Check::new(case);
        for (j, name) in names.iter().enumerate() {
            let key = format!("{stem}__{name}");
            let col = df.column(name).unwrap();
            if numeric[j] == 1.0 {
                assert!(
                    matches!(col.dtype(), DataType::Float64 | DataType::Int64),
                    "{stem}.{name}: numeric in haven, {:?} here",
                    col.dtype()
                );
                let as_f64 = col.cast(&DataType::Float64).unwrap();
                let values: Vec<f64> = as_f64
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect();
                check.vector(&key, &values, Tol::exact());
            } else {
                assert_eq!(
                    col.dtype(),
                    &DataType::String,
                    "{stem}.{name}: text in haven"
                );
                let want = case.names(&key);
                let utf8_len = case.vector(&key);
                let got: Vec<Option<&str>> = col.str().unwrap().into_iter().collect();
                assert_eq!(got.len(), want.len(), "{stem}.{name}: length");
                for (i, (g, w)) in got.iter().zip(want).enumerate() {
                    assert_eq!(w.len() as f64, utf8_len[i], "{stem}.{name}[{i}]: case JSON");
                    assert_eq!(*g, Some(w.as_str()), "{stem}.{name}[{i}]");
                }
            }
        }
        check.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::fixture_check::{assert_matches_haven, fixture_bytes, fixture_path, put};
    use super::*;
    use p2a_validation::Case;
    use std::io::Cursor;

    fn load(b: Vec<u8>) -> Result<DataFrame, SasError> {
        SasReader::new(Cursor::new(b))?.read_to_dataframe()
    }

    fn assert_invalid(r: Result<DataFrame, SasError>, needle: &str) {
        match r {
            Err(SasError::InvalidFormat(m)) => assert!(m.contains(needle), "{m}"),
            other => panic!(
                "expected InvalidFormat mentioning {needle:?}, got {:?}",
                other.map(|d| d.shape())
            ),
        }
    }

    /// The case id is an argument so that every registered test names it (strict lint,
    /// `registered-test-loads-no-case`).
    fn haven(case_id: &str, stem: &str) {
        let case = Case::load(case_id);
        let df = load_sas(fixture_path(&format!("{stem}.sas7bdat"))).unwrap();
        assert_matches_haven(&case, stem, &df);
    }

    // sas64_haven.sas7bdat (haven::write_sas: 64-bit, little-endian, no pad, header 8192, pages
    // of 4096): page 0 holds the metadata, pages 1-4 the 60 rows (19, 19, 19, 3) of 205 bytes.
    const H_PAGE0: usize = 8192;
    const H_PAGE0_SUBHEADER_COUNT: usize = H_PAGE0 + 36;
    const H_PAGE0_POINTER0: usize = H_PAGE0 + 40; // offset (i64), length (i64), compression, type
    const H_PAGE1: usize = H_PAGE0 + 4096;
    const H_PAGE1_TYPE: usize = H_PAGE1 + 32;
    const H_PAGE1_BLOCK_COUNT: usize = H_PAGE1 + 34;
    const H_ROW_SIZE: usize = H_PAGE0 + 3288; // row length at +40, row count at +48 (i64)
    const H_COL_ATTR: usize = H_PAGE0 + 2912; // vector j at +16 + 16 j: offset +0, width +8, type +14
    const H_COL_NAME: usize = H_PAGE0 + 3084; // pointer j at +16 + 8 j: text block, offset, length
    const H_COL_TEXT: usize = H_PAGE0 + 3184; // the compression literal at +8 + 12
    // sas32_iris.sas7bdat (SAS 9.4: 32-bit, little-endian, pad 4, header 65536, one mix page).
    const I_PAGE0: usize = 65536;
    const I_PAGE0_TYPE: usize = I_PAGE0 + 16;
    const I_ROW_SIZE: usize = I_PAGE0 + 65056; // row count at +24 (i32)
    const I_COL_ATTR: usize = I_PAGE0 + 64456; // vector j at +12 + 12 j: offset +0, width +4, type +10

    #[test]
    fn test_magic_number() {
        assert_eq!(SAS_MAGIC.len(), 32);
        assert_eq!(SAS_MAGIC[12], 0xc2);
    }

    #[test]
    fn test_sas_invalid_magic() {
        let result = SasReader::new(Cursor::new(
            b"Not a valid SAS file - this is just random text".to_vec(),
        ));
        match result {
            Err(SasError::InvalidFormat(msg)) => assert!(msg.contains("magic")),
            _ => panic!("Expected InvalidFormat error"),
        }
    }

    #[test]
    fn test_sas_truncated_magic() {
        assert!(SasReader::new(Cursor::new(vec![0u8; 16])).is_err());
    }

    #[test]
    fn test_sas_fixture_offsets() {
        // The constants above are this layout; a regenerated fixture with another layout fails here.
        let b = fixture_bytes("sas64_haven.sas7bdat");
        assert_eq!(
            &b[H_ROW_SIZE..H_ROW_SIZE + 8],
            &[0xF7, 0xF7, 0xF7, 0xF7, 0, 0, 0, 0]
        );
        assert_eq!(&b[H_COL_ATTR..H_COL_ATTR + 4], &[0xFC, 0xFF, 0xFF, 0xFF]);
        assert_eq!(&b[H_COL_NAME..H_COL_NAME + 4], &[0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(&b[H_COL_TEXT..H_COL_TEXT + 4], &[0xFD, 0xFF, 0xFF, 0xFF]);
        assert_eq!(
            &b[H_PAGE1_TYPE..H_PAGE1_TYPE + 6],
            &[0x00, 0x01, 19, 0, 0, 0]
        );
        let i = fixture_bytes("sas32_iris.sas7bdat");
        assert_eq!(&i[I_ROW_SIZE..I_ROW_SIZE + 4], &[0xF7, 0xF7, 0xF7, 0xF7]);
        assert_eq!(&i[I_COL_ATTR..I_COL_ATTR + 4], &[0xFC, 0xFF, 0xFF, 0xFF]);
        assert_eq!(
            &i[I_PAGE0_TYPE..I_PAGE0_TYPE + 6],
            &[0x00, 0x02, 163, 0, 13, 0]
        );
    }

    #[test]
    fn test_validate_data_readers_manifest_lists_every_fixture() {
        // DATA-50: every fixture of the manifest exists and has its shape in the case.
        let case = Case::load("data_readers_values");
        let manifest = std::fs::read_to_string(case.dataset_path()).unwrap();
        let mut n = 0;
        for line in manifest.lines().skip(1) {
            let file = line.split(',').next().unwrap().trim_matches('"');
            let stem = file.rsplit_once('.').unwrap().0;
            assert!(fixture_path(file).is_file(), "{file}");
            assert_eq!(case.vector(&format!("{stem}__shape")).len(), 2, "{stem}");
            n += 1;
        }
        assert_eq!(n, 18);
    }

    // ---- DATA-18, DATA-19, DATA-20, DATA-47: the uncompressed layout ----

    #[test]
    fn test_sas_header_fields_follow_byte_35() {
        // DATA-18: byte 35 (not byte 32) adds the 4-byte pad before header_length, page_size and
        // page_count. sas64_haven: byte 32 = 0x33, byte 35 = 0x22 (no pad); sas64_pad: the same
        // file with the pad; sas32_iris (SAS 9.4): byte 32 = 0x22, byte 35 = 0x33.
        for (name, is_64bit, want) in [
            ("sas64_haven.sas7bdat", true, (8192, 4096, 5)),
            ("sas64_pad.sas7bdat", true, (8192, 4096, 5)),
            ("sas32_iris.sas7bdat", false, (65536, 65536, 1)),
        ] {
            let r = SasReader::new(Cursor::new(fixture_bytes(name))).unwrap();
            assert_eq!(r.is_64bit, is_64bit, "{name}");
            assert_eq!((r.header_length, r.page_size, r.page_count), want, "{name}");
        }
    }

    #[test]
    fn test_sas_page_header_counts() {
        // DATA-20: block count at B + 2, subheader count at B + 4. sas64_haven's data pages hold
        // 19, 19, 19 and 3 blocks; sas32_iris's mix page 163 blocks = 13 subheaders + 150 rows.
        assert_eq!(
            load_sas(fixture_path("sas64_haven.sas7bdat"))
                .unwrap()
                .height(),
            60
        );
        assert_eq!(
            load_sas(fixture_path("sas64_pad.sas7bdat"))
                .unwrap()
                .height(),
            60
        );
        assert_eq!(
            load_sas(fixture_path("sas32_iris.sas7bdat"))
                .unwrap()
                .height(),
            150
        );
    }

    #[test]
    fn test_sas_column_types_from_the_attribute_vector() {
        // DATA-19: the type byte is at int_len + 6 of each attribute vector, after the name flag.
        use ColumnType::{Character as C, Numeric as N};
        let r = SasReader::new(Cursor::new(fixture_bytes("sas64_haven.sas7bdat"))).unwrap();
        let types: Vec<ColumnType> = r.columns.iter().map(|c| c.col_type).collect();
        assert_eq!(types, [N, N, N, N, N, N, C, C, C]);
        let r = SasReader::new(Cursor::new(fixture_bytes("sas32_iris.sas7bdat"))).unwrap();
        let types: Vec<ColumnType> = r.columns.iter().map(|c| c.col_type).collect();
        assert_eq!(types, [N, N, N, N, C]);
    }

    #[test]
    fn test_validate_sas_unknown_column_type_is_error() {
        // DATA-19: an unknown type code is an error, not Character (ReadStat refuses too:
        // guard__sas_type_code_3 = -1).
        let case = Case::load("data_readers_values");
        assert_eq!(case.scalar("guard__sas_type_code_3"), -1.0);
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        b[H_COL_ATTR + 16 + 14] = 3;
        assert_invalid(load(b), "column 0 has the unknown type code 3");
    }

    #[test]
    fn test_validate_sas_column_names_from_the_column_text() {
        // DATA-47: names come from the column-text blocks the name pointers index.
        let case = Case::load("data_readers_values");
        for stem in ["sas64_haven", "sas32_iris"] {
            let df = load_sas(fixture_path(&format!("{stem}.sas7bdat"))).unwrap();
            let names: Vec<String> = df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            assert_eq!(
                names.as_slice(),
                case.names(&format!("{stem}__columns")),
                "{stem}"
            );
        }
    }

    #[test]
    fn test_sas_name_pointer_outside_text_is_error() {
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_COL_NAME + 16 + 2, &500u16.to_le_bytes());
        assert_invalid(
            load(b),
            "the name of column 0 points outside the column text",
        );
    }

    // ---- T0 (DATA-49) guards, rebuilt on the real layout ----

    #[test]
    fn sas_rejects_zero_row_length() {
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_ROW_SIZE + 40, &0u64.to_le_bytes());
        put(&mut b, H_ROW_SIZE + 48, &100_000u64.to_le_bytes());
        put(&mut b, H_PAGE1_BLOCK_COUNT, &0xFFFFu16.to_le_bytes());
        assert_invalid(load(b), "row_length");
    }

    #[test]
    fn sas_rejects_negative_fields() {
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_COL_ATTR + 16 + 16 * 6 + 8, &(-1i32).to_le_bytes()); // grp width -1
        assert_invalid(load(b), "width");
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_PAGE0_POINTER0, &(-100i64).to_le_bytes()); // first subheader offset
        assert_invalid(load(b), "subheader offset");
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_PAGE0_SUBHEADER_COUNT, &0xFFFFu16.to_le_bytes()); // 65535 pointers
        assert_invalid(load(b), "subheader count");
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_PAGE1_BLOCK_COUNT, &0xFFFFu16.to_le_bytes());
        assert_invalid(load(b), "block count");
    }

    #[test]
    fn sas_bounds_rows_and_columns_by_the_layout() {
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_ROW_SIZE + 48, &1_000_000u64.to_le_bytes()); // > 5 pages x 19 rows
        assert_invalid(load(b), "row_count");
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_PAGE1_BLOCK_COUNT, &300u16.to_le_bytes()); // a page holds (4096 - 40) / 205 = 19
        assert_invalid(load(b), "block count 300 on page 1 exceeds the 19 rows");
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_COL_ATTR + 16 + 16 * 6 + 8, &14u32.to_le_bytes()); // grp 7 -> 14: 212 > 205
        assert_invalid(load(b), "column widths add up to 212 bytes");
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_COL_ATTR + 16 + 16, &200u64.to_le_bytes()); // income at 200, width 8
        assert_invalid(load(b), "extends past");
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_COL_ATTR + 16 + 16 + 8, &0u32.to_le_bytes()); // income width 0, inside the row (T0 da640108)
        assert_invalid(load(b), "has width 0");
    }

    #[test]
    fn test_validate_sas_stops_at_row_count_for_character_only_files() {
        // ReadStat reads the same bytes as 2 text columns' 2 rows (guard__sas_char_only_row_count_2),
        // and stops at a smaller row count too (guard__sas_row_count_59).
        let case = Case::load("data_readers_values");
        let mut b = fixture_bytes("sas32_iris.sas7bdat");
        for j in 0..5 {
            b[I_COL_ATTR + 12 + 12 * j + 10] = 2; // every column character
        }
        put(&mut b, I_ROW_SIZE + 24, &2u32.to_le_bytes());
        let df = load(b).unwrap();
        assert_eq!(df.shape(), (2, 5));
        assert_eq!(
            df.height() as f64,
            case.scalar("guard__sas_char_only_row_count_2")
        );
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_ROW_SIZE + 48, &59u64.to_le_bytes());
        assert_eq!(
            load(b).unwrap().height() as f64,
            case.scalar("guard__sas_row_count_59")
        );
    }

    // ---- DATA-21: numerics shorter than 8 bytes; every NaN is missing, nothing else ----

    #[test]
    fn test_validate_read_sas7bdat_64_bit() {
        // DATA-18, DATA-19, DATA-20, DATA-47 together; ratio[10] = 1e308 is a value (DATA-21).
        haven("data_readers_values", "sas64_haven");
    }

    #[test]
    fn test_validate_read_sas7bdat_64_bit_pad() {
        haven("data_readers_values", "sas64_pad");
    }

    #[test]
    fn test_validate_read_sas7bdat_short_numerics() {
        // DATA-21: widths 3, 4, 6, 7, 5, 8 are truncated doubles (width 4 is not an f32).
        haven("data_readers_values", "sas64_short");
    }

    #[test]
    fn test_validate_read_sas7bdat_short_numerics_big_endian() {
        // DATA-21: in a big-endian file the kept bytes lead the double.
        haven("data_readers_values", "sas64_short_be");
    }

    #[test]
    fn test_sas_truncated_numerics_both_byte_orders() {
        // 1.0 = 3FF0000000000000, 10804.37 = 40C51A2F5C28F5C3, -0.1 = BFB999999999999A. Keeping
        // the high-order bytes: 3 -> 1.0; 4 -> 40C51A2F00000000 = 10804.3671875;
        // 6 -> BFB9999999990000 = -0.0999999999994543 (haven: sas64_short ratio[11]).
        assert_eq!(
            decode_numeric(&[0x3F, 0xF0, 0x00], ByteOrder::BigEndian),
            Some(1.0)
        );
        assert_eq!(
            decode_numeric(&[0x00, 0xF0, 0x3F], ByteOrder::LittleEndian),
            Some(1.0)
        );
        assert_eq!(
            decode_numeric(&[0x40, 0xC5, 0x1A, 0x2F], ByteOrder::BigEndian),
            Some(10804.3671875)
        );
        assert_eq!(
            decode_numeric(&[0x2F, 0x1A, 0xC5, 0x40], ByteOrder::LittleEndian),
            Some(10804.3671875)
        );
        assert_eq!(
            decode_numeric(&[0xBF, 0xB9, 0x99, 0x99, 0x99, 0x99], ByteOrder::BigEndian),
            Some(-0.0999999999994543)
        );
        assert_eq!(
            decode_numeric(&1e308f64.to_le_bytes(), ByteOrder::LittleEndian),
            Some(1e308)
        );
        assert_eq!(
            decode_numeric(&[0xFF, 0xFF, 0xFE], ByteOrder::BigEndian),
            None
        ); // SAS missing
    }

    #[test]
    fn sas_rejects_wide_numerics() {
        let mut b = fixture_bytes("sas64_haven.sas7bdat");
        put(&mut b, H_COL_ATTR + 16 + 16 + 8, &9u32.to_le_bytes()); // income width 9
        assert_invalid(
            load(b),
            "numeric column income has width 9; SAS numerics are 1 to 8 bytes",
        );
    }

    // ---- text by the header's encoding byte ----

    #[test]
    fn test_validate_read_sas7bdat_windows_1252() {
        // Encoding 62 (wlatin1), as every SAS-written file checked: "\u{20ac}" is 0x80.
        haven("data_readers_values", "sas64_wlatin1");
    }

    #[test]
    fn test_validate_read_sas7bdat_encoding_codes() {
        // ReadStat on the same bytes: code 204 reads as Windows-1252 (guard__sas_wlatin1_code_204
        // = 60 rows), code 29 as ISO-8859-1 (0x80 -> U+0080), an unassigned Windows-1252 byte is
        // refused (guard__sas_wlatin1_byte_0x81 = -1).
        let case = Case::load("data_readers_values");
        let mut b = fixture_bytes("sas64_wlatin1.sas7bdat");
        b[70] = 204;
        let df = load(b).unwrap();
        assert_eq!(
            df.height() as f64,
            case.scalar("guard__sas_wlatin1_code_204")
        );
        assert_matches_haven(&case, "sas64_wlatin1", &df);
        let mut b = fixture_bytes("sas64_wlatin1.sas7bdat");
        b[70] = 29;
        let df = load(b).unwrap();
        let got: Vec<Option<&str>> = df
            .column("grp")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .collect();
        for (g, w) in got.iter().zip(case.names("sas64_wlatin1_code29__grp")) {
            assert_eq!(*g, Some(w.as_str()));
        }
        assert_eq!(case.scalar("guard__sas_wlatin1_byte_0x81"), -1.0);
        let mut b = fixture_bytes("sas64_wlatin1.sas7bdat");
        b[H_PAGE1 + 40 + 48] = 0x81; // the first byte of row 1's grp cell ("Z\u{fc}rich")
        match load(b) {
            Err(SasError::ParseError(m)) => {
                assert!(
                    m.contains("column grp, row 1: byte 0x81 is not a Windows-1252"),
                    "{m}"
                )
            }
            other => panic!("expected ParseError, got {:?}", other.map(|d| d.shape())),
        }
    }

    #[test]
    fn test_sas_text_encodings() {
        assert_eq!(
            decode_windows_1252(&[0x5A, 0xFC, 0x72, 0x80, 0x35]).unwrap(),
            "Z\u{fc}r\u{20ac}5"
        );
        assert_eq!(decode_windows_1252(&[0x41, 0x8D]), Err(0x8D));
        assert_eq!(decode_windows_1252_lenient(&[0x8D, 0x9F]), "\u{8d}\u{178}");
        assert_eq!(
            decode_utf8_or_windows_1252("Z\u{fc}rich".as_bytes()),
            "Z\u{fc}rich"
        );
        assert_eq!(decode_utf8_or_windows_1252(&[0x5A, 0xFC]), "Z\u{fc}");
        assert_eq!(decode_sas_text(&[0xE9], 29).unwrap(), "\u{e9}");
        assert_eq!(decode_sas_text(&[0x80], 62).unwrap(), "\u{20ac}");
        assert_eq!(decode_sas_text(&[0x80], 0).unwrap(), "\u{20ac}");
        assert!(
            decode_sas_text(&[0xFF], 20)
                .unwrap_err()
                .contains("not valid UTF-8")
        );
        assert!(
            decode_sas_text(&[0xE9], 28)
                .unwrap_err()
                .contains("US-ASCII")
        );
        assert!(
            decode_sas_text(&[0xA4], 64)
                .unwrap_err()
                .contains("code 64 is not supported")
        );
        assert_eq!(decode_sas_text(b"plain", 64).unwrap(), "plain");
    }
}
