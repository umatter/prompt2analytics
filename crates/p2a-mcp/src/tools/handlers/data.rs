//! Data management tool handlers.
//!
//! This module defines data loading, export, and inspection tools
//! using the `#[tool_router(router = data_router)]` pattern.
//!
//! Note: Cleaning session tools remain in server.rs due to their complexity.

use std::io::Write;
use std::path::{Path, PathBuf};

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::path_jail;
use crate::server::AnalyticsServer;
use crate::session;
use crate::tools::handlers::utils::CappedWriter;
use crate::tools::requests::data::*;

use p2a_core::data::{DataLoader, Dataset, DatasetInfo};
use p2a_core::polars::prelude::{
    AnyValue, CategoricalPhysical, Column, CsvWriter, DataFrame, DataType, ParquetReader,
    ParquetWriter, SerReader, SerWriter,
};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

/// Rows serialised between two cancellation checkpoints in a JSON export.
const CHECKPOINT_ROWS: usize = 1 << 16;

/// The file formats `export_dataset` writes.
#[derive(Clone, Copy)]
enum ExportKind {
    Csv,
    Parquet,
    Json,
}

/// Why an export could not be written.
enum ExportError {
    /// The output passed the export_bytes budget.
    TooLarge,
    /// The writer failed (the detail may still name the file).
    Failed(String),
}

fn export_limit_message(cap: Option<u64>) -> String {
    format!(
        "The export is larger than this server's export limit of {} bytes; export fewer rows or \
         columns and try again.",
        cap.unwrap_or_default()
    )
}

/// Write `df` as `kind` to the new file `tmp` through a buffered writer
/// capped at `cap` bytes.
fn write_export(
    df: &DataFrame,
    kind: ExportKind,
    tmp: &Path,
    cap: Option<u64>,
) -> Result<(), ExportError> {
    let file = std::fs::File::create(tmp).map_err(|e| ExportError::Failed(e.to_string()))?;
    let mut w = CappedWriter::new(std::io::BufWriter::new(file), cap);
    let result = match kind {
        ExportKind::Csv => CsvWriter::new(&mut w)
            .finish(&mut df.clone())
            .map_err(|e| e.to_string()),
        ExportKind::Parquet => ParquetWriter::new(&mut w)
            .finish(&mut df.clone())
            .map(|_| ())
            .map_err(|e| e.to_string()),
        ExportKind::Json => {
            serde_json::to_writer_pretty(&mut w, &RecordsJson(df)).map_err(|e| e.to_string())
        }
    };
    let result = result.and_then(|()| w.flush().map_err(|e| e.to_string()));
    if w.exceeded() {
        return Err(ExportError::TooLarge);
    }
    result.map_err(ExportError::Failed)
}

/// A data frame serialised in the records layout of
/// `Dataset::to_json_string` (an array with one object per row, keys in
/// column order) without building it in memory. Values map as there, except
/// that text cells are plain JSON strings (the legacy path formatted them
/// with polars' `Display`, which wraps them in literal quote characters).
struct RecordsJson<'a>(&'a DataFrame);

impl Serialize for RecordsJson<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::Error as _;
        let df = self.0;
        let columns = df.get_columns();
        let mut seq = serializer.serialize_seq(Some(df.height()))?;
        for row in 0..df.height() {
            if row % CHECKPOINT_ROWS == 0 {
                p2a_core::resource::checkpoint().map_err(S::Error::custom)?;
            }
            seq.serialize_element(&RecordRow { columns, row })?;
        }
        seq.end()
    }
}

/// One row of [`RecordsJson`].
struct RecordRow<'a> {
    columns: &'a [Column],
    row: usize,
}

impl Serialize for RecordRow<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.columns.len()))?;
        for col in self.columns {
            map.serialize_entry(col.name().as_str(), &CellJson(col.get(self.row).ok()))?;
        }
        map.end()
    }
}

/// One cell of [`RecordsJson`]; `None` (a failed read) is null.
struct CellJson<'a>(Option<AnyValue<'a>>);

impl Serialize for CellJson<'_> {
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
            // serde_json writes non-finite floats as null, as the legacy export did.
            Some(AnyValue::Float32(v)) => s.serialize_f64(f64::from(*v)),
            Some(AnyValue::Float64(v)) => s.serialize_f64(*v),
            Some(AnyValue::String(v)) => s.serialize_str(v),
            Some(AnyValue::StringOwned(v)) => s.serialize_str(v.as_str()),
            Some(other) => s.serialize_str(&format!("{other}")),
        }
    }
}

/// What decoding a parquet file allocates at least, read from its footer
/// and page headers before anything is decoded (see
/// [`parquet_decoded_bytes`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DecodedBound {
    /// The least the dataset quotas charge for the dataset once it is
    /// decoded.
    dataset: u64,
    /// The decoder's own buffers on top of the dataset while it decodes,
    /// freed once the file is loaded (see [`chunk_transient`]).
    transient: u64,
}

/// What decoding the parquet file at `path` allocates at least, read from
/// its footer and page headers before anything is decoded. A small file can
/// declare hundreds of millions of rows, one row holding a list of as many
/// values (one value run-length encoded), or a page that decompresses to
/// gigabytes, and the quotas only see a dataset once it is built.
///
/// The dataset part is the least the quotas will charge once the file is
/// decoded: each column's values at their in-memory width, a list's from
/// the entries counted below it (see [`least_value_bytes`]), plus the
/// dataset's structure. It is taken per row group and summed: polars
/// decodes each row group on its own (several at once), so what one group
/// declares cannot cancel what another holds. The transient part is the
/// decoder's buffers for the column chunks it decodes at once (see
/// [`chunk_transient`] and [`decode_threads`]).
///
/// The counts are the ones the decoder follows, not the footer's summary of
/// them: the rows are each row group's own (polars decodes those and never
/// reads the file-level count), and a leaf column's entries and page sizes
/// are what the page headers declare over the pages the decoder reads (see
/// [`PageWalk::chunk`]) where that is more than the column chunk's footer
/// entry says. So a footer that understates either is still charged what
/// decoding the file allocates.
///
/// `Ok(None)` when the footer cannot be read; the loader then reports the
/// file's error. `Err` says why the file is not read: its column chunks
/// overlap or pass its end (see [`check_chunk_ranges`]), or the call was
/// cancelled or passed its deadline while the page headers were read.
fn parquet_decoded_bytes(path: &Path) -> Result<Option<DecodedBound>, String> {
    let Ok(file) = std::fs::File::open(path) else {
        return Ok(None);
    };
    let mut reader = ParquetReader::new(file);
    let Ok(metadata) = reader.get_metadata().map(std::sync::Arc::clone) else {
        return Ok(None);
    };
    let Ok(schema) = reader.schema() else {
        return Ok(None);
    };
    // A second handle for the page headers: the reader keeps its own.
    let Ok(file) = std::fs::File::open(path) else {
        return Ok(None);
    };
    let Ok(file_len) = file.metadata().map(|meta| meta.len()) else {
        return Ok(None);
    };
    check_chunk_ranges(
        metadata
            .row_groups
            .iter()
            .flat_map(|group| group.byte_ranges_iter()),
        file_len,
    )?;
    let columns: Vec<(&str, DataType)> = schema
        .iter_values()
        .map(|field| (field.name.as_str(), DataType::from_arrow_field(field)))
        .collect();
    let mut walk = PageWalk::new(std::io::BufReader::new(file), file_len);
    let mut dataset = 0u64;
    let mut transients = Vec::new();
    for group in &metadata.row_groups {
        let rows = group.num_rows() as u64;
        for (name, dtype) in &columns {
            let leaf_types = leaf_types(dtype);
            let mut leaves = Vec::with_capacity(leaf_types.len());
            // The column chunks under the top-level column, in the order of
            // the file's columns, which is the order of its leaves.
            let chunks = group.columns_under_root_iter(name).into_iter().flatten();
            for (i, chunk) in chunks.enumerate() {
                let read = walk.chunk(chunk.byte_range(), chunk.num_values())?;
                leaves.push(LeafChunks {
                    entries: u64::try_from(chunk.num_values())
                        .unwrap_or(0)
                        .max(read.entries),
                    page_bytes: u64::try_from(chunk.uncompressed_size())
                        .unwrap_or(0)
                        .max(read.page_bytes),
                });
                // polars' own footer type: this stops compiling when the
                // thrift parser the walk uses is not the one polars reads
                // the file with (see Cargo.toml).
                let codec: polars_parquet_format::CompressionCodec = chunk.metadata().codec;
                transients.push(chunk_transient(
                    &read,
                    leaf_types.get(i).copied(),
                    codec != polars_parquet_format::CompressionCodec::UNCOMPRESSED,
                    dtype.is_nested(),
                ));
            }
            dataset = dataset.saturating_add(least_value_bytes(
                dtype,
                Slots::Known(rows),
                &mut leaves.into_iter(),
            ));
        }
    }
    let structure = session::structure_bytes(columns.iter().map(|(name, _)| *name));
    Ok(Some(DecodedBound {
        dataset: dataset.saturating_add(structure),
        transient: largest_sum(transients, decode_threads()),
    }))
}

/// Refuse a file of `file_len` bytes whose column chunks (their byte
/// ranges, `ranges`) pass its end or share bytes. No writer produces
/// either: polars cannot read a chunk past the end of the file, and chunks
/// that share their bytes would have those page headers read once for
/// each, so the work before the file is admitted would grow with the number
/// of chunks rather than with the file.
fn check_chunk_ranges(
    ranges: impl Iterator<Item = std::ops::Range<u64>>,
    file_len: u64,
) -> Result<(), String> {
    let mut ranges: Vec<_> = ranges.collect();
    if ranges.iter().any(|range| range.end > file_len) {
        return Err("a column chunk of the file ends past the end of the file".to_string());
    }
    ranges.retain(|range| !range.is_empty());
    ranges.sort_unstable_by_key(|range| range.start);
    if ranges.windows(2).any(|pair| pair[1].start < pair[0].end) {
        return Err("two column chunks of the file share bytes".to_string());
    }
    Ok(())
}

/// The leaf types of `dtype`, in the order of its parquet leaf columns
/// (the order [`least_value_bytes`] consumes them in).
fn leaf_types(dtype: &DataType) -> Vec<&DataType> {
    fn push<'a>(dtype: &'a DataType, out: &mut Vec<&'a DataType>) {
        match dtype {
            DataType::List(inner) | DataType::Array(inner, _) => push(inner, out),
            DataType::Struct(fields) => fields.iter().for_each(|field| push(field.dtype(), out)),
            leaf => out.push(leaf),
        }
    }
    let mut out = Vec::new();
    push(dtype, &mut out);
    out
}

/// What is known about one leaf column of one row group before decoding.
#[derive(Clone, Copy, Default)]
struct LeafChunks {
    /// Its entries: values, nulls, and empty or null lists above them.
    entries: u64,
    /// Its pages' uncompressed size.
    page_bytes: u64,
}

/// What the page headers of one column chunk declare, over the pages
/// polars' page reader reads (see [`PageWalk::chunk`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ChunkPages {
    /// The entries of its data pages.
    entries: u64,
    /// The uncompressed size of its pages, dictionary page included.
    page_bytes: u64,
    /// The uncompressed size of its largest data page.
    largest_page: u64,
    /// The entries of its data page with the most.
    most_page_entries: u64,
    /// The uncompressed size of its dictionary page.
    dict_bytes: u64,
    /// The entries of its dictionary page.
    dict_entries: u64,
}

/// Bytes of one entry's definition and repetition levels, a `u16` each, as
/// polars decodes them for a nested column's page.
const LEVEL_BYTES: u64 = 4;

/// The decoder's own buffers while it decodes one column chunk, from what
/// its page headers declare (`read`), for the leaf type `leaf` (`None` when
/// the schema has fewer leaves than the footer has columns, counted as
/// text), freed once the chunk is decoded. polars sizes each from what the
/// headers declare before it decompresses or decodes the page, so a page
/// that declares gigabytes gets them, whatever it holds:
/// - a compressed chunk's pages are decompressed into buffers of the size
///   their headers declare, zero-filled first (so resident): one for the
///   data pages, reused and grown to the largest, and one for the
///   dictionary page;
/// - the dictionary is decoded whole, whatever the rows: room for each
///   entry its header declares at the leaf's width (a view for text,
///   Categorical included) and a byte, and a text dictionary's bytes copied
///   out of its page;
/// - a nested column's pages have their levels decoded whole, into buffers
///   grown to the page with the most entries.
///
/// These count for every leaf type, fixed-width ones included, whose
/// dataset charge does not grow with their pages.
fn chunk_transient(
    read: &ChunkPages,
    leaf: Option<&DataType>,
    compressed: bool,
    nested: bool,
) -> u64 {
    let decompressed = if compressed {
        read.largest_page.saturating_add(read.dict_bytes)
    } else {
        0
    };
    let (entry, text) = match leaf {
        None
        | Some(
            DataType::String
            | DataType::Binary
            | DataType::BinaryOffset
            | DataType::Categorical(_, _)
            | DataType::Enum(_, _),
        ) => (session::VIEW_BYTES, true),
        // polars decodes no dictionary for a Boolean column.
        Some(DataType::Boolean) => (0, false),
        Some(other) => (leaf_value_bytes(other, 1), false),
    };
    let dictionary = entry
        .saturating_add(1)
        .saturating_mul(read.dict_entries)
        .saturating_add(if text { read.dict_bytes } else { 0 });
    let levels = if nested {
        LEVEL_BYTES.saturating_mul(read.most_page_entries)
    } else {
        0
    };
    decompressed
        .saturating_add(dictionary)
        .saturating_add(levels)
}

/// The sum of the `k` largest of `values`.
fn largest_sum(mut values: Vec<u64>, k: usize) -> u64 {
    values.sort_unstable_by(|a, b| b.cmp(a));
    values.into_iter().take(k).fold(0u64, u64::saturating_add)
}

/// The most column chunks polars decodes at once: the threads of its pool,
/// sized as polars sizes it (`POLARS_MAX_THREADS`, else the available
/// parallelism). Each thread decodes one chunk at a time, whether polars
/// spreads a file's row groups or its columns over them.
fn decode_threads() -> usize {
    std::env::var("POLARS_MAX_THREADS")
        .ok()
        .and_then(|threads| threads.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Page headers parsed between two cancellation checkpoints.
const CHECKPOINT_HEADERS: u64 = 1 << 10;

/// Reads the page headers of a parquet file's column chunks, as polars'
/// page reader (`PageReader`) does before it decodes a chunk, without
/// decompressing or decoding anything.
struct PageWalk {
    file: std::io::BufReader<std::fs::File>,
    file_len: u64,
    /// Headers parsed so far.
    headers: u64,
}

impl PageWalk {
    fn new(file: std::io::BufReader<std::fs::File>, file_len: u64) -> Self {
        Self {
            file,
            file_len,
            headers: 0,
        }
    }

    /// Move to `offset` in the file, keeping what is buffered when it is
    /// there (a file's chunks follow one another).
    fn seek_to(&mut self, offset: u64) -> bool {
        use std::io::Seek;
        let Ok(at) = self.file.stream_position() else {
            return false;
        };
        let (Ok(offset), Ok(at)) = (i64::try_from(offset), i64::try_from(at)) else {
            return false;
        };
        self.file.seek_relative(offset - at).is_ok()
    }

    /// Parse the next page header from no more than `left` bytes, taking
    /// its size off `left`; `Ok(None)` when it does not parse. The reader
    /// parses with no size limit, from the chunk's bytes alone; the limit of
    /// eight per byte left refuses no header the reader accepts (the
    /// protocol counts at most eight per byte it reads) and bounds what a
    /// header can make the parser reserve. `Err` when the call is cancelled
    /// or past its deadline.
    fn header(
        &mut self,
        left: &mut u64,
    ) -> Result<Option<polars_parquet_format::PageHeader>, String> {
        use polars_parquet_format::thrift::protocol::TCompactInputProtocol;
        use std::io::Read;
        self.headers += 1;
        if self.headers.is_multiple_of(CHECKPOINT_HEADERS) {
            p2a_core::resource::checkpoint()
                .map_err(|e| format!("the call stopped while the file was checked ({e})"))?;
        }
        let mut chunk = (&mut self.file).take(*left);
        let limit = usize::try_from(left.saturating_mul(8)).unwrap_or(usize::MAX);
        let header = polars_parquet_format::PageHeader::read_from_in_protocol(
            &mut TCompactInputProtocol::new(&mut chunk, limit),
        );
        *left = chunk.limit();
        Ok(header.ok())
    }

    /// Skip a page of `size` bytes (at most `left`) after its header.
    fn skip(&mut self, size: u64, left: &mut u64) -> bool {
        // `size` is at most `left`, which came from two u64 file offsets
        // the footer stores as i64.
        if size > *left {
            return false;
        }
        let Ok(skip) = i64::try_from(size) else {
            return false;
        };
        if self.file.seek_relative(skip).is_err() {
            return false;
        }
        *left -= size;
        true
    }

    /// Read the page headers of the column chunk at `range` as polars' page
    /// reader does. First a dictionary page, which the reader reads whatever
    /// the chunk's count says (`read_dict`); then, after it or from the
    /// chunk's start when the first page is not one, a data page header and
    /// its page at a time, while the data pages' declared entries sum to
    /// less than `declared` (the chunk's footer entry, signed as the reader
    /// keeps it; the reader counts a whole page at a time, so the last page
    /// it reads can hold far more). It stops where the reader fails: a
    /// header that does not parse, a page type it does not read, a
    /// dictionary page past the first, a page without its type's header or
    /// with a negative size, or a page past the chunk's end. `Err` when the
    /// call is cancelled or past its deadline.
    fn chunk(&mut self, range: std::ops::Range<u64>, declared: i64) -> Result<ChunkPages, String> {
        use polars_parquet_format::PageType;
        let mut read = ChunkPages::default();
        let end = range.end.min(self.file_len);
        if range.start >= end || !self.seek_to(range.start) {
            return Ok(read);
        }
        let mut left = end - range.start;
        let Some(first) = self.header(&mut left)? else {
            return Ok(read);
        };
        if first.type_ == PageType::DICTIONARY_PAGE {
            let entries = first
                .dictionary_page_header
                .as_ref()
                .map(|dict| u64::try_from(dict.num_values));
            let size = u64::try_from(first.compressed_page_size);
            let bytes = u64::try_from(first.uncompressed_page_size);
            let (Some(Ok(entries)), Ok(size), Ok(bytes)) = (entries, size, bytes) else {
                return Ok(read);
            };
            if !self.skip(size, &mut left) {
                return Ok(read);
            }
            read.dict_bytes = bytes;
            read.dict_entries = entries;
            read.page_bytes = bytes;
        } else if self.seek_to(range.start) {
            left = end - range.start;
        } else {
            return Ok(read);
        }
        // The reader's own tally, which a negative page count lowers.
        let mut seen = 0i64;
        while seen < declared {
            let Some(header) = self.header(&mut left)? else {
                break;
            };
            let entries = match header.type_ {
                PageType::DATA_PAGE => header.data_page_header.as_ref().map(|h| h.num_values),
                PageType::DATA_PAGE_V2 => header.data_page_header_v2.as_ref().map(|h| h.num_values),
                _ => None,
            };
            let size = u64::try_from(header.compressed_page_size);
            let bytes = u64::try_from(header.uncompressed_page_size);
            let (Some(entries), Ok(size), Ok(bytes)) = (entries, size, bytes) else {
                break;
            };
            if !self.skip(size, &mut left) {
                break;
            }
            seen = seen.saturating_add(i64::from(entries));
            let entries = u64::try_from(entries).unwrap_or(0);
            read.entries = read.entries.saturating_add(entries);
            read.page_bytes = read.page_bytes.saturating_add(bytes);
            read.largest_page = read.largest_page.max(bytes);
            read.most_page_entries = read.most_page_entries.max(entries);
        }
        Ok(read)
    }
}

/// How many values a column, or a part of a nested one, holds, for
/// [`least_value_bytes`].
#[derive(Clone, Copy)]
enum Slots {
    /// No list above: the rows, times the sizes of the arrays above. Each
    /// leaf column below also takes room for the entries it declares where
    /// they are more: polars reserves room for a nested column chunk's
    /// footer count before it decodes it, and an honest chunk declares no
    /// more entries than this.
    Known(u64),
    /// Below an array of size zero: no values, whatever the entries.
    Empty,
    /// Below a list: unknown, but each entry counted for a leaf column below
    /// is a value of it, an empty or null list below the outermost one, or
    /// one of the outermost list's slots, whose number `outer` holds.
    /// `nested` is set below a second list, where an empty inner list is an
    /// entry of every leaf column below it.
    BelowList { outer: u64, nested: bool },
}

/// Bytes of a list's offset: the least an empty list below another one
/// takes.
const OFFSET_BYTES: u64 = 8;

/// The least memory the values of `dtype` take in polars, for `slots` of it,
/// from what the footer and page headers say about each leaf column
/// (`leaves`, consumed in order, one per leaf; see [`parquet_decoded_bytes`]).
///
/// A leaf column's value takes its physical width (a view per String or
/// Binary value, a code per Categorical one); a list its offsets, plus what
/// its values take. A list's values are counted from the entries counted
/// for each leaf column below it: past the outermost list's slots (one
/// entry each at most when empty or null), each entry is a value or an empty
/// list below, which take the leaf's width or an offset, so the least of the
/// two is counted. Below two lists an empty inner list is an entry of each
/// leaf column of a struct inside it, though it takes one offset, so the
/// sum over the struct's fields would count it once per field; such a
/// struct counts the most any one of its fields does instead, which is
/// still a lower bound (and still counts a field that holds many values).
/// A text or categorical leaf takes at least its pages (long
/// strings, a dictionary of long labels): the quotas charge more for each
/// value and label than its pages hold. A fixed-width leaf's charge does not
/// grow with its pages, which also hold levels its values in memory do not;
/// what decoding its pages allocates is counted with the decoder's buffers
/// instead (see [`chunk_transient`]). Types without a fixed floor count
/// nothing.
fn least_value_bytes(
    dtype: &DataType,
    slots: Slots,
    leaves: &mut impl Iterator<Item = LeafChunks>,
) -> u64 {
    match dtype {
        DataType::List(inner) => match slots {
            // An outermost list in no slots holds nothing but what its leaf
            // columns declare.
            Slots::Known(0) | Slots::Empty => least_value_bytes(inner, slots, leaves),
            Slots::Known(n) => OFFSET_BYTES
                .saturating_mul(n)
                .saturating_add(least_value_bytes(
                    inner,
                    Slots::BelowList {
                        outer: n,
                        nested: false,
                    },
                    leaves,
                )),
            Slots::BelowList { outer, .. } => least_value_bytes(
                inner,
                Slots::BelowList {
                    outer,
                    nested: true,
                },
                leaves,
            ),
        },
        DataType::Array(inner, size) => match slots {
            // An array of size zero holds no values; an empty or null one is
            // an entry without them.
            _ if *size == 0 => least_value_bytes(inner, Slots::Empty, leaves),
            Slots::Known(n) => {
                least_value_bytes(inner, Slots::Known(n.saturating_mul(*size as u64)), leaves)
            }
            Slots::Empty | Slots::BelowList { .. } => least_value_bytes(inner, slots, leaves),
        },
        DataType::Struct(fields) => {
            // Every field consumes its own leaves, whichever bound is kept.
            let bounds = fields
                .iter()
                .map(|field| least_value_bytes(field.dtype(), slots, leaves));
            match slots {
                Slots::BelowList { nested: true, .. } => bounds.fold(0u64, u64::max),
                _ => bounds.fold(0u64, u64::saturating_add),
            }
        }
        leaf => {
            let chunks = leaves.next().unwrap_or_default();
            let values = match slots {
                Slots::Empty => return 0,
                Slots::Known(n) => leaf_value_bytes(leaf, n.max(chunks.entries)),
                Slots::BelowList { outer, .. } => {
                    let values = chunks.entries.saturating_sub(outer);
                    leaf_value_bytes(leaf, values).min(OFFSET_BYTES.saturating_mul(values))
                }
            };
            let text = matches!(
                leaf,
                DataType::String
                    | DataType::Binary
                    | DataType::BinaryOffset
                    | DataType::Categorical(_, _)
                    | DataType::Enum(_, _)
            );
            if text {
                values.max(chunks.page_bytes)
            } else {
                values
            }
        }
    }
}

/// The least memory `values` values of the leaf (not nested) type `dtype`
/// take in polars: its physical width. Types without a fixed floor count
/// nothing.
fn leaf_value_bytes(dtype: &DataType, values: u64) -> u64 {
    let width = |bytes: u64| bytes.saturating_mul(values);
    match dtype {
        DataType::Boolean => values.div_ceil(8),
        DataType::Int8 | DataType::UInt8 => width(1),
        DataType::Int16 | DataType::UInt16 => width(2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date => width(4),
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Datetime(_, _)
        | DataType::Duration(_)
        | DataType::Time
        | DataType::BinaryOffset => width(8),
        DataType::Int128 | DataType::Decimal(_, _) => width(16),
        DataType::String | DataType::Binary => width(session::VIEW_BYTES),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => match dtype.cat_physical() {
            Ok(CategoricalPhysical::U8) => width(1),
            Ok(CategoricalPhysical::U16) => width(2),
            Ok(CategoricalPhysical::U32) => width(4),
            Err(_) => 0,
        },
        _ => 0,
    }
}

/// Why a parquet file is not loaded, decided before it is decoded.
enum NotAdmitted {
    /// The dataset quotas refuse it: the handler's error.
    Quota(McpError),
    /// It is not read: the reason, for the tool's error result.
    Unread(String),
}

/// Refuse, before decoding it, a parquet file at `path` that the dataset
/// quotas would refuse once it was loaded as `name` (see
/// [`session::admit_dataset`]), counting what decoding it allocates on top
/// of the dataset (see [`parquet_decoded_bytes`]). Other formats, and calls
/// without quotas, pass: a CSV, Excel, Stata or SAS file is charged only
/// once decoded, and a CSV can decode to several times its size on disk. A
/// file whose footer cannot be read passes too (the loader reports its
/// error); one the check will not read (see [`parquet_decoded_bytes`]) is
/// not loaded, and neither is one whose schema makes the check panic:
/// polars' schema conversion (`DataType::from_arrow_field`) panics on some
/// malformed categorical metadata, and the loader runs the same code.
async fn admit_parquet(path: &Path, name: &str) -> Result<(), NotAdmitted> {
    let is_parquet = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_lowercase().as_str(), "parquet" | "pq"));
    if !is_parquet || !session::dataset_admission_active() {
        return Ok(());
    }
    let bound = match std::panic::catch_unwind(|| parquet_decoded_bytes(path)) {
        Ok(Ok(Some(bound))) => bound,
        Ok(Ok(None)) => return Ok(()),
        Ok(Err(reason)) => return Err(NotAdmitted::Unread(reason)),
        Err(_) => {
            tracing::info!("the parquet check panicked reading a file's schema");
            return Err(NotAdmitted::Unread(
                "could not read the file's schema".to_string(),
            ));
        }
    };
    session::admit_dataset(name, bound.dataset.saturating_add(bound.transient))
        .await
        .map_err(|refusal| NotAdmitted::Quota(session::quota_refusal(refusal)))
}

/// A file removed when this drops, also when the call unwinds.
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `text` (an error from a loader or writer) with the absolute forms of
/// `path` and its folder replaced by their [`path_jail::display_path`] forms,
/// so no server path reaches the caller.
fn without_server_paths(text: &str, path: &Path) -> String {
    let mut out = text.replace(&path.display().to_string(), &path_jail::display_path(path));
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        out = out.replace(
            &parent.display().to_string(),
            &path_jail::display_path(parent),
        );
    }
    out
}

#[tool_router(router = data_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Dataset Loading/Saving Tools
    // ========================================================================

    /// List all currently loaded datasets.
    #[tool(
        description = "List all currently loaded datasets with their basic information (name, dimensions, column types)."
    )]
    pub async fn list_datasets(&self) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        if datasets.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No datasets currently loaded. Use the 'load_dataset' tool to load a data file.",
            )]));
        }

        let mut result = String::from("Loaded Datasets:\n\n");
        for (id, dataset) in datasets.iter() {
            let info: DatasetInfo = dataset.into();
            result.push_str(&format!(
                "- **{}**: {} rows x {} columns\n",
                id, info.nrows, info.ncols
            ));
            result.push_str("  Columns: ");
            let col_summary: Vec<String> = info
                .columns
                .iter()
                .map(|c| format!("{} ({})", c.name, c.dtype))
                .collect();
            result.push_str(&col_summary.join(", "));
            result.push_str("\n\n");
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Load a dataset from a file.
    #[tool(
        description = "Load a dataset from a file. Supports CSV, Parquet, Excel (xlsx, xls, xlsb, ods), Stata (dta), and SAS (sas7bdat) formats. Returns dataset information including dimensions and column types."
    )]
    pub async fn load_dataset(
        &self,
        Parameters(request): Parameters<LoadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let path = match path_jail::validate_data_path(&request.path) {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Refused to load dataset: {}",
                    e
                ))]));
            }
        };

        // The configurable maximum file size: checked on disk before any
        // parsing, only when the limits set a load_file_bytes budget (hosted
        // 64 MiB, none when limits are off, P2A_LIMIT_OVERRIDES in any mode).
        let limits = crate::tools::limits::current();
        if limits.load_file_bytes.is_some() {
            let size = match std::fs::metadata(&path) {
                Ok(meta) => meta.len(),
                Err(_) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to load dataset: File not found: {}",
                        path_jail::display_path(&path)
                    ))]));
                }
            };
            if let Err(v) = limits.check_budget(
                "load_dataset",
                "load_file_bytes",
                size,
                limits.load_file_bytes,
            ) {
                tracing::info!(
                    tool = "load_dataset",
                    override_key = v.override_key,
                    requested = %v.requested,
                    limit = %v.limit,
                    "request refused by a named budget"
                );
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "{} Load a smaller file, or split it and load the parts.",
                    v.message
                ))]));
            }
        }

        // Generate an ID for the dataset
        let id = request.name.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset")
                .to_string()
        });

        // A parquet file the dataset quotas would refuse is refused before
        // it is decoded.
        match admit_parquet(&path, &id).await {
            Ok(()) => {}
            Err(NotAdmitted::Quota(refusal)) => return Err(refusal),
            Err(NotAdmitted::Unread(reason)) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to load dataset: {reason}. Nothing was loaded."
                ))]));
            }
        }

        // Load the dataset
        let dataset = match DataLoader::load(&path) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to load dataset: {}",
                    without_server_paths(&e.to_string(), &path)
                ))]));
            }
        };

        // Get info before moving
        let info: DatasetInfo = (&dataset).into();

        // Track memory usage
        {
            let mut profiler = self.memory_profiler.write().await;
            profiler.track_dataset(&id, &dataset);
        }

        // Store the dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(id.clone(), dataset);

        let result = format!(
            "Successfully loaded dataset '{}'\n\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            id,
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!(
                    "  - {} ({}): {} nulls ({:.1}%)",
                    c.name,
                    c.dtype,
                    c.null_count,
                    if info.nrows > 0 {
                        (c.null_count as f64 / info.nrows as f64) * 100.0
                    } else {
                        0.0
                    }
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Export a dataset to a file.
    #[tool(
        description = "Export a loaded dataset to a file. Supports CSV, Parquet, and JSON formats. The format is determined by the file extension or can be explicitly specified."
    )]
    pub async fn export_dataset(
        &self,
        Parameters(request): Parameters<ExportDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let path = match path_jail::validate_data_path_for_write(&request.path) {
            Ok(p) => p,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Refused to export dataset: {}",
                    e
                ))]));
            }
        };

        // Determine format from extension or explicit parameter
        let format = request
            .format
            .as_deref()
            .or_else(|| path.extension().and_then(|ext| ext.to_str()));

        let (kind, label) = match format {
            Some("csv") => (ExportKind::Csv, "CSV"),
            Some("parquet") => (ExportKind::Parquet, "Parquet"),
            Some("json") => (ExportKind::Json, "JSON"),
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unsupported format '{}'. Use 'csv', 'parquet', or 'json'.",
                    other
                ))]));
            }
            None => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Could not determine output format. Please specify a format or use a file extension.",
                )]));
            }
        };

        // Written to a temporary file in the validated folder, capped at the
        // export_bytes budget, then renamed into place (path_jail checks the
        // session storage quota before the rename).
        let cap = crate::tools::limits::current().export_bytes;
        let written = path_jail::write_atomically(&path, |tmp| {
            write_export(dataset.df(), kind, tmp, cap).map_err(|e| match e {
                ExportError::TooLarge => export_limit_message(cap),
                ExportError::Failed(detail) => format!(
                    "Failed to export to {label}: {}",
                    without_server_paths(&without_server_paths(&detail, tmp), &path)
                ),
            })
        });
        if let Err(e) = written {
            return Ok(CallToolResult::error(vec![Content::text(e)]));
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Successfully exported dataset '{}' to {label} file:\n  {}\n  {} rows x {} columns",
            request.dataset,
            path_jail::display_path(&path),
            dataset.nrows(),
            dataset.ncols()
        ))]))
    }

    /// Upload and load a dataset from base64-encoded content.
    #[tool(
        description = "Upload and load a dataset from base64-encoded content. Useful when the client cannot access the filesystem directly. Supports CSV and Parquet formats."
    )]
    pub async fn upload_dataset(
        &self,
        Parameters(request): Parameters<UploadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        use base64::Engine;

        // Decode base64
        let decoded = match base64::engine::general_purpose::STANDARD.decode(&request.content) {
            Ok(d) => d,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to decode base64 content: {}",
                    e
                ))]));
            }
        };

        // Determine format from filename extension
        let path = PathBuf::from(&request.filename);
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("csv");

        // Generate ID
        let id = request.name.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("uploaded")
                .to_string()
        });

        // Load based on format
        let dataset = match extension.to_lowercase().as_str() {
            "csv" => {
                // Convert bytes to string and use from_csv_string
                let csv_str = match String::from_utf8(decoded.clone()) {
                    Ok(s) => s,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to decode CSV as UTF-8: {}",
                            e
                        ))]));
                    }
                };
                match DataLoader::from_csv_string(&csv_str) {
                    Ok(df) => Dataset::new(df),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse CSV: {}",
                            e
                        ))]));
                    }
                }
            }
            "parquet" => {
                // Written to a new temporary file and loaded from there
                // (Parquet is a binary format); the file is removed however
                // the call ends, a panic in the check or the loader
                // included.
                let temp_path =
                    std::env::temp_dir().join(format!("upload_{}.parquet", uuid::Uuid::new_v4()));
                let created = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&temp_path);
                let temp = match created {
                    Ok(mut file) => {
                        let temp = RemoveOnDrop(temp_path);
                        if let Err(e) = file.write_all(&decoded) {
                            return Ok(CallToolResult::error(vec![Content::text(format!(
                                "Failed to write temp file: {}",
                                e
                            ))]));
                        }
                        temp
                    }
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to write temp file: {}",
                            e
                        ))]));
                    }
                };

                // Refused before decoding when the quotas would refuse the
                // decoded dataset.
                match admit_parquet(&temp.0, &id).await {
                    Ok(()) => {}
                    Err(NotAdmitted::Quota(refusal)) => return Err(refusal),
                    Err(NotAdmitted::Unread(reason)) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse Parquet: {reason}. Nothing was loaded."
                        ))]));
                    }
                }
                let result = DataLoader::load(&temp.0);
                drop(temp);

                match result {
                    Ok(ds) => ds,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse Parquet: {}",
                            e
                        ))]));
                    }
                }
            }
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unsupported file format '{}'. Upload supports CSV and Parquet only.",
                    other
                ))]));
            }
        };

        // Get info
        let info: DatasetInfo = (&dataset).into();

        // Store
        let mut datasets = self.datasets.write().await;
        datasets.insert(id.clone(), dataset);

        let result = format!(
            "Successfully uploaded dataset '{}'\n\n\
             Source: {} ({} bytes decoded)\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            id,
            request.filename,
            decoded.len(),
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!("  - {} ({})", c.name, c.dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Create a dataset from inline CSV content.
    #[tool(
        description = "Create a dataset directly from CSV content provided as a string. Useful for creating small datasets without file I/O."
    )]
    pub async fn create_dataset(
        &self,
        Parameters(request): Parameters<CreateDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        // from_csv_string returns DataFrame, wrap in Dataset
        let df = match DataLoader::from_csv_string(&request.csv_content) {
            Ok(df) => df,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to parse CSV content: {}",
                    e
                ))]));
            }
        };
        let dataset = Dataset::new(df);

        let info: DatasetInfo = (&dataset).into();

        let mut datasets = self.datasets.write().await;
        datasets.insert(request.name.clone(), dataset);

        let result = format!(
            "Successfully created dataset '{}'\n\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            request.name,
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!("  - {} ({})", c.name, c.dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ========================================================================
    // Dataset Inspection Tools
    // ========================================================================

    /// Describe a loaded dataset with summary statistics.
    #[tool(
        description = "Get summary statistics for a loaded dataset including column types, null counts, and basic statistics for numeric columns."
    )]
    pub async fn describe_dataset(
        &self,
        Parameters(request): Parameters<DescribeDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let info: DatasetInfo = dataset.into();
        let result = format!(
            "Dataset: {}\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            request.dataset,
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!(
                    "  - {} ({}): {} nulls ({:.1}%)",
                    c.name,
                    c.dtype,
                    c.null_count,
                    if info.nrows > 0 {
                        (c.null_count as f64 / info.nrows as f64) * 100.0
                    } else {
                        0.0
                    }
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Preview the first rows of a dataset.
    #[tool(
        description = "Preview the first N rows of a loaded dataset (default: 5 rows). Returns data in a formatted table."
    )]
    pub async fn head_dataset(
        &self,
        Parameters(request): Parameters<HeadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

        let n = request.n.unwrap_or(5);
        let df = dataset.df();
        let preview = df.head(Some(n));

        let result = format!(
            "Dataset: {} (showing first {} of {} rows)\n\n{}",
            request.dataset,
            n.min(df.height()),
            df.height(),
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_core::polars::prelude::df;
    use serde_json::json;

    /// [`least_value_bytes`] of a whole column of `rows` rows whose leaf
    /// columns have `entries` entries, and `page_bytes` of pages each.
    fn least_with_pages(dtype: &DataType, rows: u64, entries: &[u64], page_bytes: u64) -> u64 {
        let mut leaves = entries.iter().map(|&entries| LeafChunks {
            entries,
            page_bytes,
        });
        least_value_bytes(dtype, Slots::Known(rows), &mut leaves)
    }

    fn least(dtype: &DataType, rows: u64, entries: &[u64]) -> u64 {
        least_with_pages(dtype, rows, entries, 0)
    }

    fn list(inner: DataType) -> DataType {
        DataType::List(Box::new(inner))
    }

    fn fields(fields: &[(&str, DataType)]) -> DataType {
        DataType::Struct(
            fields
                .iter()
                .map(|(name, dtype)| {
                    p2a_core::polars::prelude::Field::new((*name).into(), dtype.clone())
                })
                .collect(),
        )
    }

    #[test]
    fn a_lists_values_are_counted_from_the_footers_entries() {
        // One row holding a list of 20,000,000 floats: its offsets and every
        // value past the row's own slot. Pre-fix a list column counted 8
        // bytes a row, whatever its lists held.
        assert_eq!(
            least(&list(DataType::Float64), 1, &[20_000_000]),
            8 + 8 * (20_000_000 - 1)
        );
        // Below the outermost list a value takes its width, or an offset
        // where that is less (an entry can be an empty inner list).
        assert_eq!(least(&list(list(DataType::Int32)), 2, &[10]), 16 + 4 * 8);
        assert_eq!(least(&list(DataType::String), 1, &[5]), 8 + 8 * 4);
        assert_eq!(least(&list(DataType::Boolean), 1, &[17]), 8 + 2);
        // Each leaf column of a struct in a list has its own entries.
        let pairs = list(fields(&[("a", DataType::Float64), ("b", DataType::UInt8)]));
        assert_eq!(least(&pairs, 3, &[10, 10]), 24 + 8 * 7 + 7);
        // Arrays outside a list have the slots their size gives.
        let array = DataType::Array(Box::new(DataType::Float64), 3);
        assert_eq!(least(&array, 2, &[6]), 48);
        assert_eq!(least(&list(array), 2, &[9]), 16 + 8 * 7);
        // An array of size zero holds no values, in a list or not.
        let empty = DataType::Array(Box::new(DataType::Float64), 0);
        assert_eq!(least(&empty, 4, &[4]), 0);
        assert_eq!(least(&list(empty), 1, &[5]), 8);
        // Below two lists an empty inner list is an entry of every field of
        // a struct in it but takes one offset, so the struct counts its
        // largest field rather than the sum (which counted 8 + 32 + 32 + 4).
        let deep = list(list(fields(&[
            ("a", DataType::Float64),
            ("b", DataType::Float64),
            ("c", DataType::UInt8),
        ])));
        assert_eq!(least(&deep, 1, &[5, 5, 5]), 8 + 8 * 4);
        assert_eq!(least(&deep, 1, &[5, 9, 5]), 8 + 8 * 8);
        // A struct in one list still sums its fields, and so does a struct
        // above the inner list.
        let above = list(fields(&[
            ("l", list(DataType::Float64)),
            ("f", DataType::Float64),
        ]));
        assert_eq!(least(&above, 1, &[5, 3]), 8 + 8 * 4 + 8 * 2);
        // A struct beside lists: each field consumes its own leaves.
        let mixed = fields(&[
            ("l", list(DataType::Int64)),
            ("f", DataType::Float32),
            ("m", list(DataType::Int16)),
        ]);
        assert_eq!(
            least(&mixed, 2, &[12, 2, 6]),
            (16 + 8 * 10) + 8 + (16 + 2 * 4)
        );
        // A footer with fewer entries than slots, or none, counts no values.
        assert_eq!(least(&list(DataType::Float64), 5, &[3]), 40);
        assert_eq!(least(&list(DataType::Float64), 5, &[]), 40);
        // Flat columns count their rows, as before.
        assert_eq!(least(&DataType::Float64, 7, &[7]), 56);
        assert_eq!(least(&DataType::String, 2, &[2]), 32);
        // Outside a list, a leaf column takes room for the entries it
        // declares where they are more than its slots: polars reserves room
        // for a nested chunk's footer count before it decodes it. Pre-fix a
        // struct column counted its rows only.
        let pair = fields(&[("a", DataType::Float64), ("b", DataType::String)]);
        assert_eq!(least(&pair, 1, &[1_000_000, 3]), 8_000_000 + 16 * 3);
        assert_eq!(least(&pair, 4, &[4, 4]), 32 + 64);
        assert_eq!(least(&list(DataType::Float64), 0, &[9]), 72);
        // A text or categorical leaf counts its pages where they are more
        // (long strings, a dictionary of long labels). A fixed-width leaf's
        // pages count with the decoder's buffers instead (see the test of
        // `chunk_transient` below), not in the dataset's charge.
        assert_eq!(least_with_pages(&DataType::String, 2, &[2], 1000), 1000);
        assert_eq!(
            least_with_pages(&list(DataType::Binary), 1, &[3], 1000),
            8 + 1000
        );
        assert_eq!(least_with_pages(&DataType::String, 200, &[200], 1000), 3200);
    }

    #[test]
    fn the_decoders_buffers_count_for_every_leaf_type() {
        // Pre-fix a fixed-width column's pages counted nothing, though
        // polars decompresses each into a zero-filled buffer of the size
        // its header declares.
        let pages = ChunkPages {
            entries: 2,
            page_bytes: 3000,
            largest_page: 1000,
            most_page_entries: 2,
            ..ChunkPages::default()
        };
        let f64 = Some(&DataType::Float64);
        assert_eq!(chunk_transient(&pages, f64, true, false), 1000);
        // Uncompressed pages are decoded where they lie in the file.
        assert_eq!(chunk_transient(&pages, f64, false, false), 0);
        // A dictionary page: its decompression buffer, and its entries
        // decoded at their width (and a byte each).
        let dict = ChunkPages {
            dict_bytes: 5000,
            dict_entries: 600,
            ..pages
        };
        assert_eq!(
            chunk_transient(&dict, f64, true, false),
            1000 + 5000 + 9 * 600
        );
        assert_eq!(chunk_transient(&dict, f64, false, false), 9 * 600);
        // A text dictionary is decoded into views, its bytes copied out.
        let text = Some(&DataType::String);
        assert_eq!(chunk_transient(&dict, text, false, false), 17 * 600 + 5000);
        assert_eq!(chunk_transient(&dict, None, false, false), 17 * 600 + 5000);
        // A nested column's levels, two u16 per entry of its largest page.
        let levels = ChunkPages {
            most_page_entries: 1 << 20,
            ..ChunkPages::default()
        };
        let flag = Some(&DataType::Boolean);
        assert_eq!(chunk_transient(&levels, flag, false, true), 4 << 20);
        assert_eq!(chunk_transient(&levels, flag, false, false), 0);
        // The chunks decoded at once: the largest, as many as threads.
        assert_eq!(largest_sum(vec![5, 1, 9, 3], 2), 14);
        assert_eq!(largest_sum(vec![5, 1], 8), 6);
        assert!(decode_threads() >= 1);
    }

    /// What decoding the parquet file at `path` allocates at least: its
    /// dataset and transient bounds.
    fn bound_of(path: &Path) -> DecodedBound {
        parquet_decoded_bytes(path).unwrap().unwrap()
    }

    /// What the quotas charge for `df` once written to parquet (in row
    /// groups of `rows_per_group` rows, or polars' default) and loaded as
    /// `d`, and the least [`parquet_decoded_bytes`] gave for its dataset
    /// before.
    fn loaded_charge_and_bound(df: &mut DataFrame, rows_per_group: Option<usize>) -> (u64, u64) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.parquet");
        p2a_core::polars::prelude::ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .with_row_group_size(rows_per_group)
            .finish(df)
            .unwrap();
        let bound = bound_of(&path).dataset;
        let loaded = DataLoader::load(&path).unwrap();
        assert_eq!(loaded.df().dtypes(), df.dtypes());
        let charge = session::datasets_bytes_of([("d", &loaded)]);
        (charge, bound)
    }

    #[test]
    fn the_bound_before_decoding_stays_below_what_the_quotas_charge_after() {
        use p2a_core::polars::prelude::{IntoSeries, ListChunked, NamedFrom, Series};
        // Lists with nulls, empty lists and null values, a list of lists
        // with empty inner lists, a list of structs, a list of lists of
        // structs with empty inner lists, an array and a list of
        // categoricals: the check before decoding must not refuse a file the
        // quotas would hold once it is loaded, and must count the values the
        // lists hold, not 8 bytes a row.
        let floats = |n: usize, shift: f64| -> Series {
            Series::new(
                "".into(),
                (0..n)
                    .map(|i| Some(i as f64 * 0.37 + shift))
                    .collect::<Vec<_>>(),
            )
        };
        let rows = 2_000usize;
        let lists: ListChunked = (0..rows)
            .map(|i| match i % 7 {
                0 => None,
                1 => Some(Series::new_empty("".into(), &DataType::Float64)),
                2 => Some(Series::new("".into(), [Some(0.5), None, Some(1.25)])),
                k => Some(floats(k * 5, i as f64 * 0.01)),
            })
            .collect();
        let nested: ListChunked = (0..rows)
            .map(|i| {
                let inner: ListChunked = (0..(i % 4))
                    .map(|j| match (i + j) % 3 {
                        0 => Some(Series::new_empty("".into(), &DataType::Int32)),
                        k => Some(Series::new("".into(), vec![(i * 3 + j) as i32 + 1; k * 2])),
                    })
                    .collect();
                Some(inner.into_series().cast(&list(DataType::Int32)).unwrap())
            })
            .collect();
        let structs: ListChunked = (0..rows)
            .map(|i| {
                let n = i % 5;
                let a = floats(n, 0.3);
                let b = Series::new(
                    "b".into(),
                    (0..n)
                        .map(|j| format!("s{}", i * 7 + j))
                        .collect::<Vec<_>>(),
                );
                let s = p2a_core::polars::prelude::StructChunked::from_series(
                    "".into(),
                    n,
                    [a.with_name("a".into()), b].iter(),
                )
                .unwrap()
                .into_series();
                Some(s)
            })
            .collect();
        // Lists of lists of a struct of 20 floats, a third of them holding
        // five empty inner lists: the parquet leaf of every field counts
        // each empty inner list, which takes one offset.
        let struct_of = |n: usize, seed: usize| -> Series {
            let fields: Vec<Series> = (0..20)
                .map(|f| floats(n, (seed * 7 + f) as f64 * 0.1).with_name(format!("f{f}").into()))
                .collect();
            p2a_core::polars::prelude::StructChunked::from_series("".into(), n, fields.iter())
                .unwrap()
                .into_series()
        };
        let deep: ListChunked = (0..rows)
            .map(|i| {
                let inner: ListChunked = if i % 3 == 0 {
                    (0..5).map(|_| Some(struct_of(0, i))).collect()
                } else {
                    std::iter::once(Some(struct_of(i % 4, i))).collect()
                };
                Some(inner.into_series())
            })
            .collect();
        let cat = DataType::from_categories(p2a_core::polars::prelude::Categories::random(
            "".into(),
            CategoricalPhysical::U32,
        ));
        let cats: ListChunked = (0..rows)
            .map(|i| {
                let labels: Vec<String> = (0..(i % 6))
                    .map(|j| format!("lab{}", (i + j) % 11))
                    .collect();
                Some(Series::new("".into(), labels).cast(&cat).unwrap())
            })
            .collect();
        let array = Series::new("arr".into(), floats(rows * 3, 0.9))
            .reshape_array(&[
                p2a_core::polars::prelude::ReshapeDimension::Infer,
                p2a_core::polars::prelude::ReshapeDimension::new_dimension(3),
            ])
            .unwrap();
        let mut df = DataFrame::new(vec![
            lists.into_series().with_name("lists".into()).into(),
            nested.into_series().with_name("nested".into()).into(),
            structs.into_series().with_name("structs".into()).into(),
            cats.into_series().with_name("cats".into()).into(),
            deep.into_series().with_name("deep".into()).into(),
            array.into(),
            floats(rows, 0.1).with_name("flat".into()).into(),
        ])
        .unwrap();
        let inner_floats = df
            .column("lists")
            .unwrap()
            .list()
            .unwrap()
            .get_inner()
            .len() as u64;
        // In one row group and in several: the bound is summed over the
        // row groups.
        for groups in [None, Some(300)] {
            let (charge, bound) = loaded_charge_and_bound(&mut df, groups);
            assert!(
                bound <= charge,
                "{groups:?}: bound {bound} > charge {charge}"
            );
            assert!(
                bound >= 8 * inner_floats,
                "{groups:?}: bound {bound}, {inner_floats} list values"
            );

            // Each column alone, too.
            for name in ["lists", "nested", "structs", "cats", "deep", "arr"] {
                let mut one = df.select([name]).unwrap();
                let (charge, bound) = loaded_charge_and_bound(&mut one, groups);
                assert!(
                    bound <= charge,
                    "{groups:?} {name}: bound {bound} > charge {charge}"
                );
            }
        }
    }

    /// Rewrite the footer of the parquet file at `path` with `edit`, leaving
    /// its pages where they are.
    fn rewrite_footer(path: &Path, edit: impl FnOnce(&mut polars_parquet_format::FileMetaData)) {
        use polars_parquet_format::thrift::protocol::{
            TCompactInputProtocol, TCompactOutputProtocol,
        };
        let bytes = std::fs::read(path).unwrap();
        let n = bytes.len();
        assert_eq!(&bytes[n - 4..], b"PAR1");
        let len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
        let start = n - 8 - len;
        let mut input = TCompactInputProtocol::new(&bytes[start..n - 8], usize::MAX);
        let mut footer =
            polars_parquet_format::FileMetaData::read_from_in_protocol(&mut input).unwrap();
        edit(&mut footer);
        let mut written = Vec::new();
        footer
            .write_to_out_protocol(&mut TCompactOutputProtocol::new(&mut written))
            .unwrap();
        let mut out = bytes[..start].to_vec();
        out.extend_from_slice(&written);
        out.extend_from_slice(&u32::try_from(written.len()).unwrap().to_le_bytes());
        out.extend_from_slice(b"PAR1");
        std::fs::write(path, out).unwrap();
    }

    /// Write `df` as a parquet file of `rows_per_group` rows per row group
    /// at `path`.
    fn write_parquet(df: &mut DataFrame, path: &Path, rows_per_group: usize) {
        p2a_core::polars::prelude::ParquetWriter::new(std::fs::File::create(path).unwrap())
            .with_row_group_size(Some(rows_per_group))
            .finish(df)
            .unwrap();
    }

    #[test]
    fn a_footer_that_understates_rows_or_entries_does_not_lower_the_bound() {
        use p2a_core::polars::prelude::{NamedFrom, Series};
        let dir = tempfile::tempdir().unwrap();
        // Rows: polars decodes each row group's own count, never the
        // file-level one, which the bound used to take.
        let flat = dir.path().join("flat.parquet");
        let values: Vec<f64> = (0..300_000).map(|i| (i % 89) as f64 * 0.31 + 0.2).collect();
        let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
        write_parquet(&mut df, &flat, 100_000);
        let honest = bound_of(&flat);
        assert!(honest.dataset >= 8 * 300_000, "{honest:?}");
        rewrite_footer(&flat, |footer| footer.num_rows = 1);
        assert_eq!(bound_of(&flat), honest);
        // Entries: a chunk is decoded a page at a time until the pages have
        // given the chunk's count, so a chunk that says it holds one entry
        // is still decoded through its first page, here all 500,000 values
        // of one row's list.
        let lists = dir.path().join("lists.parquet");
        let inner = Series::new(
            "".into(),
            (0..500_000)
                .map(|i| (i % 13) as f64 + 0.5)
                .collect::<Vec<_>>(),
        );
        let mut df = DataFrame::new(vec![Series::new("l".into(), [inner]).into()]).unwrap();
        write_parquet(&mut df, &lists, 1);
        let honest = bound_of(&lists).dataset;
        assert!(honest >= 8 * 499_999, "{honest}");
        rewrite_footer(&lists, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                let meta = chunk.meta_data.as_mut().unwrap();
                meta.num_values = 1;
                // A footer can understate the pages' size too.
                meta.total_uncompressed_size = 1;
            }
        });
        assert_eq!(bound_of(&lists).dataset, honest);
        // Pages: a text column is charged at least its pages uncompressed,
        // which a footer can understate; the page headers say what the
        // decoder decompresses.
        let text = dir.path().join("text.parquet");
        let long = Series::new(
            "s".into(),
            (0..2_000).map(|i| format!("{i:01000}")).collect::<Vec<_>>(),
        );
        let mut df = DataFrame::new(vec![long.into()]).unwrap();
        write_parquet(&mut df, &text, 2_000);
        rewrite_footer(&text, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                chunk.meta_data.as_mut().unwrap().total_uncompressed_size = 1;
            }
        });
        let bound = bound_of(&text).dataset;
        assert!(bound >= 2_000 * 1_000, "{bound}");
    }

    #[test]
    fn a_struct_chunk_that_declares_more_entries_counts_them() {
        // polars reserves room for a nested column chunk's footer count, at
        // the leaf's width, before it decodes the chunk (`collect_nested`);
        // a one-row struct column whose chunk declares billions of entries
        // makes that reservation fail, which aborts the process. Pre-fix
        // the bound counted the struct's rows only.
        use p2a_core::polars::prelude::{IntoSeries, NamedFrom, Series, StructChunked};
        let a = Series::new("a".into(), [0.5f64, 1.75]);
        let s = StructChunked::from_series("s".into(), 2, [a].iter())
            .unwrap()
            .into_series();
        let mut df = DataFrame::new(vec![s.into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("struct.parquet");
        write_parquet(&mut df, &path, 2);
        assert!(bound_of(&path).dataset < 1 << 20);
        rewrite_footer(&path, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                chunk.meta_data.as_mut().unwrap().num_values = 1 << 31;
            }
        });
        let bound = bound_of(&path).dataset;
        assert!(bound >= 8 << 31, "{bound}");
    }

    #[test]
    fn the_page_headers_give_each_chunk_the_entries_its_footer_does() {
        // On a file polars wrote, the walk over the page headers ends where
        // the chunk's footer count says, with the same count, for flat,
        // nested and dictionary-encoded columns over several row groups.
        use p2a_core::polars::prelude::{NamedFrom, Series};
        let rows = 30_000usize;
        let floats = Series::new(
            "f".into(),
            (0..rows)
                .map(|i| (i % 5 != 0).then_some(i as f64 * 0.7 + 0.1))
                .collect::<Vec<_>>(),
        );
        let text = Series::new(
            "s".into(),
            (0..rows)
                .map(|i| format!("v{}", i % 300))
                .collect::<Vec<_>>(),
        );
        let lists = Series::new(
            "l".into(),
            (0..rows)
                .map(|i| Series::new("".into(), vec![i as i32 + 1; i % 4]))
                .collect::<Vec<_>>(),
        );
        let mut df = DataFrame::new(vec![floats.into(), text.into(), lists.into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.parquet");
        write_parquet(&mut df, &path, 7_000);
        let metadata = std::sync::Arc::clone(
            ParquetReader::new(std::fs::File::open(&path).unwrap())
                .get_metadata()
                .unwrap(),
        );
        let mut walk = walk_of(&path);
        let mut chunks = 0;
        let mut dictionaries = 0;
        for group in &metadata.row_groups {
            for chunk in group.parquet_columns() {
                let read = walk.chunk(chunk.byte_range(), chunk.num_values()).unwrap();
                assert_eq!(read.entries, chunk.num_values() as u64);
                assert!(read.page_bytes > 0);
                assert!(read.page_bytes <= chunk.uncompressed_size() as u64);
                assert!(read.largest_page > 0 && read.largest_page <= read.page_bytes);
                assert!(read.most_page_entries > 0 && read.most_page_entries <= read.entries);
                assert!(read.dict_bytes <= read.page_bytes);
                if read.dict_bytes > 0 {
                    assert!(read.dict_entries > 0);
                    dictionaries += 1;
                }
                chunks += 1;
            }
        }
        assert!(metadata.row_groups.len() > 1);
        assert_eq!(chunks, metadata.row_groups.len() * 3);
        assert!(dictionaries >= metadata.row_groups.len(), "{dictionaries}");
    }

    /// A [`PageWalk`] over the file at `path`.
    fn walk_of(path: &Path) -> PageWalk {
        let file = std::fs::File::open(path).unwrap();
        let file_len = file.metadata().unwrap().len();
        PageWalk::new(std::io::BufReader::new(file), file_len)
    }

    /// A one-column parquet file at `path` whose column `k` is a dictionary
    /// of `labels` distinct labels of 1,000 bytes, one row each, in one row
    /// group: it loads as Categorical.
    fn long_labels_parquet(path: &Path, labels: usize) {
        use p2a_core::polars::prelude::{Categories, NamedFrom, Series};
        let values: Vec<String> = (0..labels).map(|i| format!("k{i:0999}")).collect();
        let dtype =
            DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        let k = Series::new("k".into(), values).cast(&dtype).unwrap();
        let mut df = DataFrame::new(vec![k.into()]).unwrap();
        write_parquet(&mut df, path, labels);
    }

    #[test]
    fn a_dictionary_page_counts_whatever_the_chunks_count_says() {
        // polars reads a chunk's dictionary page before it looks at the
        // chunk's count (`PageReader::read_dict`), and decodes it whole.
        // Pre-fix a chunk whose footer entry said it held no entries had no
        // page read, and its footer size counted instead.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("labels.parquet");
        long_labels_parquet(&path, 2_000);
        let honest = bound_of(&path);
        assert!(honest.dataset >= 2_000 * 1_000, "{honest:?}");
        for declared in [0, -5] {
            rewrite_footer(&path, |footer| {
                for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                    let meta = chunk.meta_data.as_mut().unwrap();
                    meta.num_values = declared;
                    meta.total_uncompressed_size = 1;
                }
            });
            let bound = bound_of(&path);
            assert!(bound.dataset >= 2_000 * 1_000, "{declared}: {bound:?}");
            // Its decoded labels and their views, too.
            assert!(
                bound.transient >= 2_000 * (1_000 + 16),
                "{declared}: {bound:?}"
            );
        }
    }

    #[test]
    fn one_row_group_cannot_cancel_what_another_holds() {
        // Two row groups: the first one row with an empty list, its footer
        // rewritten to claim 200,000 rows, the second one row holding a list
        // of 20,000 structs of 20 floats. polars decodes each group on its
        // own, the second in full (3.2 MB). Pre-fix the bound took the
        // entries and rows over the whole file, so the first group's rows
        // cancelled the second group's values: 8 bytes of offsets a claimed
        // row, and nothing for the floats.
        use p2a_core::polars::prelude::{IntoSeries, ListChunked, NamedFrom, Series};
        let fields: Vec<Series> = (0..20)
            .map(|f| {
                Series::new(
                    format!("f{f}").into(),
                    (0..20_000)
                        .map(|i| (i * 3 + f) as f64 * 0.25 + 0.1)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let wide =
            p2a_core::polars::prelude::StructChunked::from_series("".into(), 20_000, fields.iter())
                .unwrap()
                .into_series();
        let empty = wide.clear();
        let lists: ListChunked = [Some(empty), Some(wide)].into_iter().collect();
        let mut df =
            DataFrame::new(vec![lists.into_series().with_name("l".into()).into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("groups.parquet");
        write_parquet(&mut df, &path, 1);
        let honest = bound_of(&path).dataset;
        assert!(honest >= 20 * 8 * 19_999, "{honest}");
        rewrite_footer(&path, |footer| {
            assert_eq!(footer.row_groups.len(), 2);
            footer.row_groups[0].num_rows = 200_000;
            footer.num_rows = 200_001;
        });
        let bound = bound_of(&path).dataset;
        assert!(bound >= 20 * 8 * 19_999, "{bound}");
    }

    #[test]
    fn the_header_walk_stops_when_the_call_does() {
        // 3,000 row groups of one row: more than a few thousand headers,
        // so the walk reaches its checkpoints. Pre-fix it had none.
        use p2a_core::polars::prelude::{NamedFrom, Series};
        use p2a_core::resource::{Budget, CancelToken, with_budget};
        let values: Vec<f64> = (0..3_000).map(|i| i as f64 * 0.5 + 0.25).collect();
        let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("many.parquet");
        write_parquet(&mut df, &path, 1);
        let token = CancelToken::new();
        token.cancel();
        let stopped = with_budget(Budget::new(None, token), || parquet_decoded_bytes(&path));
        let reason = stopped.unwrap_err();
        assert!(reason.contains("stopped"), "{reason}");
        assert!(parquet_decoded_bytes(&path).unwrap().is_some());
    }

    #[test]
    fn column_chunks_that_share_bytes_or_pass_the_end_are_not_read() {
        // A footer can point any number of chunks at the same bytes, and
        // the walk would read their headers once per chunk; no writer does.
        use p2a_core::polars::prelude::{NamedFrom, Series};
        let values: Vec<f64> = (0..4_000).map(|i| (i % 31) as f64 * 0.5 + 0.25).collect();
        let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.parquet");
        write_parquet(&mut df, &path, 1_000);
        assert!(parquet_decoded_bytes(&path).is_ok());
        let first = |footer: &polars_parquet_format::FileMetaData| {
            footer.row_groups[0].columns[0].meta_data.clone().unwrap()
        };
        let original = std::fs::read(&path).unwrap();
        rewrite_footer(&path, |footer| {
            let shared = first(footer);
            for group in footer.row_groups.iter_mut().skip(1) {
                group.columns[0].meta_data = Some(shared.clone());
            }
        });
        let reason = parquet_decoded_bytes(&path).unwrap_err();
        assert!(reason.contains("share bytes"), "{reason}");
        std::fs::write(&path, &original).unwrap();
        rewrite_footer(&path, |footer| {
            let meta = footer.row_groups[3].columns[0].meta_data.as_mut().unwrap();
            meta.total_compressed_size += 1 << 20;
        });
        let reason = parquet_decoded_bytes(&path).unwrap_err();
        assert!(reason.contains("past the end"), "{reason}");
        // Adjacent chunks are not shared bytes.
        assert!(check_chunk_ranges([0..4, 4..9, 9..9, 12..20].into_iter(), 20).is_ok());
        assert!(check_chunk_ranges([0..4, 3..9].into_iter(), 20).is_err());
        assert!(check_chunk_ranges([0..4, 9..9, 2..3].into_iter(), 20).is_err());
        assert!(check_chunk_ranges(std::iter::once(0..21), 20).is_err());
    }

    #[test]
    fn path_schemas_say_relative_to_data_directory() {
        for schema in [
            serde_json::to_string(&schemars::schema_for!(LoadDatasetRequest)).unwrap(),
            serde_json::to_string(&schemars::schema_for!(ExportDatasetRequest)).unwrap(),
        ] {
            assert!(
                schema.contains("relative to the server's data directory"),
                "{schema}"
            );
        }
    }

    #[test]
    fn records_json_matches_to_json_string_except_string_quoting() {
        let df = df! {
            "f" => [Some(1.5), Some(f64::NAN), None],
            "i" => [Some(1i64), None, Some(3)],
            "s" => [Some("a"), None, Some("c\"q")],
            "b" => [Some(true), Some(false), None],
            "g" => [Some(0.1f32), None, Some(2.0f32)]
        }
        .unwrap();
        let mut legacy: serde_json::Value =
            serde_json::from_str(&Dataset::new(df.clone()).to_json_string().unwrap()).unwrap();
        let streamed: serde_json::Value =
            serde_json::from_slice(&serde_json::to_vec_pretty(&RecordsJson(&df)).unwrap()).unwrap();
        // Strings: plain JSON strings now (legacy wrapped them in literal quotes).
        let s: Vec<serde_json::Value> = streamed
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["s"].clone())
            .collect();
        assert_eq!(s, vec![json!("a"), json!(null), json!("c\"q")]);
        // Everything else: identical to the legacy export.
        let rows_old = legacy.as_array_mut().unwrap();
        let rows_new = streamed.as_array().unwrap();
        assert_eq!(rows_new.len(), rows_old.len());
        for (row_new, row_old) in rows_new.iter().zip(rows_old) {
            let mut row_new = row_new.clone();
            row_new.as_object_mut().unwrap().remove("s");
            row_old.as_object_mut().unwrap().remove("s");
            assert_eq!(&row_new, row_old);
        }
    }
}
