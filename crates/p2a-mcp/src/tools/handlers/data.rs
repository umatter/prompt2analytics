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
use p2a_core::polars::io::mmap::MmapBytesReader;
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
    /// The decoder's own buffers on top of the dataset while it decodes:
    /// what it decompresses and decodes into, freed once the file is
    /// loaded, and the room it reserves for a nested chunk's declared
    /// entries (see [`chunk_transient`]).
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
/// declares cannot cancel what another holds. The labels of a Categorical
/// column are the exception: polars keeps one mapping a column, so a label
/// that several row groups' dictionaries hold is held once, and each
/// distinct label of the column's dictionary pages is counted once, at what
/// the quotas charge for it, as is each value of a page that is not
/// dictionary-encoded (see [`DictionaryLabels`]). An Enum column's labels
/// are the categories its type declares, counted once a column (see
/// [`session::frozen_label_bytes`]). The transient part is the decoder's
/// buffers for the column chunks it decodes at once (see
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
/// file's error. `Err` says why the file is not read: its footer declares
/// what no writer produces (see [`read_footer`]), or the call was cancelled
/// or passed its deadline while the page headers or dictionaries were read.
fn parquet_decoded_bytes(path: &Path) -> Result<Option<DecodedBound>, String> {
    parquet_decoded_bytes_within(ParquetSource::File(path), &mut DictionaryBudget::default())
}

/// [`parquet_decoded_bytes`] of the parquet file `source`, reading no more
/// dictionary pages than `budget` allows.
fn parquet_decoded_bytes_within(
    source: ParquetSource<'_>,
    budget: &mut DictionaryBudget,
) -> Result<Option<DecodedBound>, String> {
    match read_footer(source)? {
        Some(footer) => footer_decoded_bytes(source, footer, budget),
        None => Ok(None),
    }
}

/// Where a parquet file is read from: a file of the data directory, or the
/// bytes of an upload, which are read where they lie in memory and never
/// written to disk.
#[derive(Clone, Copy)]
enum ParquetSource<'a> {
    File(&'a Path),
    Bytes(&'a [u8]),
}

/// A reader over a [`ParquetSource`].
type SourceReader<'a> = Box<dyn MmapBytesReader + 'a>;

impl<'a> ParquetSource<'a> {
    /// A new reader over the file, and its length in bytes; `None` when the
    /// file cannot be opened or measured.
    fn open(self) -> Option<(SourceReader<'a>, u64)> {
        match self {
            Self::File(path) => {
                let file = std::fs::File::open(path).ok()?;
                let len = file.metadata().ok()?.len();
                Some((Box::new(file), len))
            }
            Self::Bytes(bytes) => {
                let len = u64::try_from(bytes.len()).ok()?;
                Some((Box::new(std::io::Cursor::new(bytes)), len))
            }
        }
    }
}

/// A parquet file's footer, read and checked by [`read_footer`].
struct Footer<'a> {
    reader: ParquetReader<SourceReader<'a>>,
    /// polars' own footer type, which polars-io's `FileMetadataRef` names:
    /// this stops compiling when polars reads files with another
    /// polars-parquet than the one the check decompresses dictionaries with
    /// (see Cargo.toml).
    metadata: std::sync::Arc<polars_parquet::parquet::metadata::FileMetadata>,
    file_len: u64,
}

/// Read the footer of the parquet file `source` and refuse a file whose
/// footer declares what no writer produces and polars' reader does not
/// survive or should not be made to read: column chunks that overlap or
/// pass the file's end (see [`check_chunk_ranges`]), or one that declares a
/// negative number of values (see [`check_chunk_counts`]). Both read the
/// footer alone, so they run for every parquet load, with quotas or
/// without (see [`admit_parquet_source`]). `Ok(None)` when the footer
/// cannot be read; the loader then reports the file's error.
fn read_footer(source: ParquetSource<'_>) -> Result<Option<Footer<'_>>, String> {
    let Some((file, file_len)) = source.open() else {
        return Ok(None);
    };
    let mut reader = ParquetReader::new(file);
    let metadata: std::sync::Arc<polars_parquet::parquet::metadata::FileMetadata> =
        match reader.get_metadata() {
            Ok(metadata) => std::sync::Arc::clone(metadata),
            Err(_) => return Ok(None),
        };
    check_chunk_ranges(
        metadata
            .row_groups
            .iter()
            .flat_map(|group| group.byte_ranges_iter()),
        file_len,
    )?;
    check_chunk_counts(
        metadata
            .row_groups
            .iter()
            .flat_map(|group| group.parquet_columns().iter().map(|c| c.num_values())),
    )?;
    Ok(Some(Footer {
        reader,
        metadata,
        file_len,
    }))
}

/// [`parquet_decoded_bytes_within`] of the file `source`, whose footer
/// [`read_footer`] read.
fn footer_decoded_bytes(
    source: ParquetSource<'_>,
    footer: Footer<'_>,
    budget: &mut DictionaryBudget,
) -> Result<Option<DecodedBound>, String> {
    let Footer {
        mut reader,
        metadata,
        file_len,
    } = footer;
    let Ok(schema) = reader.schema() else {
        return Ok(None);
    };
    // A second handle for the page headers: the reader keeps its own.
    let Some((file, _)) = source.open() else {
        return Ok(None);
    };
    let columns: Vec<(&str, DataType, Vec<LeafDecode>)> = schema
        .iter_values()
        .map(|field| {
            let dtype = DataType::from_arrow_field(field);
            let mut decodes = Vec::new();
            leaf_decodes(&dtype, Some(field), &mut decodes);
            (field.name.as_str(), dtype, decodes)
        })
        .collect();
    let mut walk = PageWalk::new(std::io::BufReader::new(file), file_len);
    // An Enum leaf's labels are the categories its type declares, which
    // polars builds from the file's schema whatever its pages hold.
    let mut dataset = columns
        .iter()
        .flat_map(|(_, dtype, _)| leaf_types(dtype))
        .fold(0u64, |bytes, leaf| match leaf {
            DataType::Enum(frozen, _) => bytes.saturating_add(session::frozen_label_bytes(frozen)),
            _ => bytes,
        });
    let mut transients = Vec::new();
    // The labels of each leaf column's dictionaries over the row groups, by
    // column and leaf; only Categorical leaves add to them.
    let mut labels: Vec<Vec<DictionaryLabels>> = columns
        .iter()
        .map(|(_, dtype, _)| {
            std::iter::repeat_with(DictionaryLabels::default)
                .take(leaf_types(dtype).len())
                .collect()
        })
        .collect();
    let hasher = std::hash::RandomState::new();
    // In one row group each label has one dictionary page: its labels count
    // from its size (see `page_label_bytes`), and reading them would gain
    // no more than the rounding of their halves.
    let mut none = DictionaryBudget::none();
    let budget = if metadata.row_groups.len() > 1 {
        budget
    } else {
        &mut none
    };
    for group in &metadata.row_groups {
        let rows = group.num_rows() as u64;
        for ((name, dtype, decodes), labels) in columns.iter().zip(&mut labels) {
            let leaf_types = leaf_types(dtype);
            let mut leaves = Vec::with_capacity(leaf_types.len());
            let mut pages = Vec::with_capacity(leaf_types.len());
            // The column chunks under the top-level column, in the order of
            // the file's columns, which is the order of its leaves.
            let chunks = group.columns_under_root_iter(name).into_iter().flatten();
            for (i, chunk) in chunks.enumerate() {
                let read = walk.chunk(chunk.byte_range(), chunk.num_values())?;
                // Not negative: check_chunk_counts refused the file.
                let declared = u64::try_from(chunk.num_values()).unwrap_or(0);
                leaves.push(LeafChunks {
                    entries: declared.max(read.entries),
                    page_bytes: u64::try_from(chunk.uncompressed_size())
                        .unwrap_or(0)
                        .max(read.page_bytes),
                    data_page_bytes: read.page_bytes.saturating_sub(read.dict_bytes),
                });
                // polars' own footer type: this stops compiling when the
                // thrift parser the walk uses is not the one polars reads
                // the file with (see Cargo.toml).
                let codec: polars_parquet_format::CompressionCodec = chunk.metadata().codec;
                transients.push(chunk_transient(
                    &read,
                    leaf_types.get(i).copied(),
                    decodes.get(i).copied().unwrap_or_default(),
                    codec != polars_parquet_format::CompressionCodec::UNCOMPRESSED,
                    dtype.is_nested(),
                    declared,
                ));
                pages.push((read, codec));
            }
            let mut holds_labels = Vec::with_capacity(leaf_types.len());
            dataset = dataset.saturating_add(least_value_bytes(
                dtype,
                Slots::Known(rows),
                &mut leaves.into_iter(),
                &mut holds_labels,
            ));
            let counted = holds_labels.iter().zip(&pages).zip(labels.iter_mut());
            for ((&holds, (read, codec)), labels) in counted {
                if holds {
                    labels.plain_values = labels.plain_values.saturating_add(read.plain_values);
                    walk.dictionary(read, *codec, &hasher, labels, budget)?;
                }
            }
        }
    }
    let labels = labels
        .iter()
        .flatten()
        .fold(0u64, |bytes, labels| bytes.saturating_add(labels.bytes()));
    let structure = session::structure_bytes(columns.iter().map(|(name, _, _)| *name));
    Ok(Some(DecodedBound {
        dataset: dataset.saturating_add(labels).saturating_add(structure),
        transient: largest_sum(transients, decode_threads()),
    }))
}

/// Refuse a file a column chunk of which declares a negative number of
/// values (`counts`, the chunks' footer entries). No writer produces one:
/// polars' page reader converts the count to an unsigned size as it is
/// (`PageReader::total_num_values`), and before it decodes a nested
/// column's chunk reserves room for that many values, nearly `usize::MAX`
/// of them.
fn check_chunk_counts(mut counts: impl Iterator<Item = i64>) -> Result<(), String> {
    if counts.any(|count| count < 0) {
        return Err("a column chunk of the file declares a negative number of values".to_string());
    }
    Ok(())
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

/// The field metadata keys by which polars marks a dictionary column as its
/// own Categorical or Enum (polars-arrow's `DTYPE_ENUM_VALUES_LEGACY`,
/// `DTYPE_ENUM_VALUES_NEW`, `DTYPE_CATEGORICAL_LEGACY` and
/// `DTYPE_CATEGORICAL_NEW`), which polars-parquet's decoder looks for.
const POLARS_DICTIONARY_KEYS: [&str; 4] = [
    "_PL_ENUM_VALUES",
    "_PL_ENUM_VALUES2",
    "_PL_CATEGORICAL",
    "_PL_CATEGORICAL2",
];

/// How polars decodes the values of a Categorical or Enum leaf column of a
/// parquet file (polars-parquet's `deserialize/simple.rs`), read from its
/// arrow field. Its values become an arrow dictionary, keys and labels,
/// which polars then turns into the column's codes, so each value has a key
/// while the chunk decodes, and, for a column of another writer, a view too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LeafDecode {
    /// Bytes of one key of that dictionary: the width of the field's key
    /// type.
    key_bytes: u64,
    /// Whether the values are decoded into string views first and then cast
    /// to the dictionary: the field lacks polars' metadata (see
    /// [`POLARS_DICTIONARY_KEYS`]), as a dictionary<string> column that R's
    /// or Python's arrow writes does, or its key or label type is not one
    /// polars' categorical decoder takes.
    views: bool,
}

impl Default for LeafDecode {
    /// The least: a leaf whose field is not known.
    fn default() -> Self {
        Self {
            key_bytes: 1,
            views: false,
        }
    }
}

impl LeafDecode {
    /// The decoding of a leaf whose arrow field is `field`.
    fn of(field: &p2a_core::polars::prelude::ArrowField) -> Self {
        use p2a_core::polars::prelude::ArrowDataType as A;
        let A::Dictionary(key, labels, _) = &field.dtype else {
            return Self::default();
        };
        let key = A::from(*key);
        let key_bytes = match key {
            A::Int8 | A::UInt8 => 1,
            A::Int16 | A::UInt16 => 2,
            A::Int32 | A::UInt32 => 4,
            A::Int128 | A::UInt128 => 16,
            _ => 8,
        };
        let marked = field
            .metadata
            .as_deref()
            .is_some_and(|md| POLARS_DICTIONARY_KEYS.iter().any(|k| md.contains_key(*k)));
        let decoded = marked
            && matches!(key, A::UInt8 | A::UInt16 | A::UInt32)
            && matches!(**labels, A::Utf8View);
        Self {
            key_bytes,
            views: !decoded,
        }
    }

    /// Bytes each value takes while its chunk decodes: its key, and its view
    /// when it is decoded into one.
    fn value_bytes(self) -> u64 {
        if self.views {
            self.key_bytes.saturating_add(session::VIEW_BYTES)
        } else {
            self.key_bytes
        }
    }
}

/// Push the [`LeafDecode`] of each leaf column of `dtype`, polars' type of
/// the arrow field `field`, to `out`, in the order of [`leaf_types`]. A leaf
/// whose field the walk does not find (it follows polars' conversion of
/// lists, maps, arrays and structs) gets the least.
fn leaf_decodes(
    dtype: &DataType,
    field: Option<&p2a_core::polars::prelude::ArrowField>,
    out: &mut Vec<LeafDecode>,
) {
    use p2a_core::polars::prelude::ArrowDataType as A;
    let arrow = field.map(|field| &field.dtype);
    match dtype {
        DataType::List(inner) => {
            let inner_field = match arrow {
                Some(A::List(f) | A::LargeList(f) | A::Map(f, _)) => Some(f.as_ref()),
                _ => None,
            };
            leaf_decodes(inner, inner_field, out);
        }
        DataType::Array(inner, _) => {
            let inner_field = match arrow {
                Some(A::FixedSizeList(f, _)) => Some(f.as_ref()),
                _ => None,
            };
            leaf_decodes(inner, inner_field, out);
        }
        DataType::Struct(fields) => {
            let arrow_fields = match arrow {
                Some(A::Struct(arrow_fields)) if arrow_fields.len() == fields.len() => {
                    Some(arrow_fields)
                }
                _ => None,
            };
            for (i, f) in fields.iter().enumerate() {
                leaf_decodes(f.dtype(), arrow_fields.and_then(|fs| fs.get(i)), out);
            }
        }
        _ => out.push(field.map_or_else(LeafDecode::default, LeafDecode::of)),
    }
}

/// What is known about one leaf column of one row group before decoding.
#[derive(Clone, Copy, Default)]
struct LeafChunks {
    /// Its entries: values, nulls, and empty or null lists above them.
    entries: u64,
    /// Its pages' uncompressed size: what their headers declare (see
    /// [`ChunkPages::page_bytes`]), or the footer's size of the chunk where
    /// that is more (it counts the page headers too).
    page_bytes: u64,
    /// Its data pages' uncompressed size, as their headers declare: the
    /// pages alone, without the dictionary page or the headers (which can
    /// hold a page's least and greatest value, two labels).
    data_page_bytes: u64,
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
    /// Where its dictionary page's (compressed) bytes start in the file,
    /// after the page header, when it has one.
    dict_at: Option<u64>,
    /// The size of its dictionary page's bytes in the file.
    dict_size: u64,
    /// The values of its data pages that are not dictionary-encoded, at
    /// most: a page's entries (less the nulls a v2 header counts), and no
    /// more than a quarter of a PLAIN page's bytes (a PLAIN byte array
    /// takes four bytes of length and its text).
    plain_values: u64,
}

/// Bytes of one entry's definition and repetition levels, a `u16` each, as
/// polars decodes them for a nested column's page.
const LEVEL_BYTES: u64 = 4;

/// The decoder's own buffers while it decodes one column chunk, from what
/// its page headers declare (`read`) and its footer entry declares
/// (`declared`), for the leaf type `leaf` (`None` when the schema has fewer
/// leaves than the footer has columns, counted as text). polars sizes each
/// from what the headers or the footer declare before it decompresses or
/// decodes a page, so a chunk that declares gigabytes gets them, whatever
/// it holds:
/// - a compressed chunk's pages are decompressed into buffers of the size
///   their headers declare, zero-filled first (so resident): one for the
///   data pages, reused and grown to the largest, and one for the
///   dictionary page;
/// - the dictionary is decoded whole, whatever the rows: room for each
///   entry its header declares at the leaf's width (a view for text,
///   Categorical included), a text dictionary's bytes copied out of its
///   page, and the byte per declared entry that reading the dictionary page
///   reserves for every leaf type (`BasicDecompressor::read_dict_page`'s
///   `Vec::with_capacity`);
/// - a nested column's pages have their levels decoded whole, into buffers
///   grown to the page with the most entries;
/// - a nested column's chunk has room reserved for the entries its footer
///   declares, at the leaf's width and a validity bit each, before any of
///   its pages is decoded (`collect_nested`), and its values grow to the
///   entries its pages declare where those are more. That room becomes the
///   decoded chunk, but it is reserved whatever the column holds once
///   loaded (an array of size zero holds no values), so it counts here
///   rather than in the dataset's part. A flat column's reservation is
///   bounded by its rows, which the dataset's part counts;
/// - a Categorical or Enum leaf's values, flat or nested, are decoded into
///   an arrow dictionary before they become the column's codes: a key per
///   entry, and a view per entry first when the column is not polars' own
///   (see [`LeafDecode`], `decode`). This is the leaf's decoded chunk, so
///   it takes the place of a nested chunk's reservation above.
///
/// These count for every leaf type, fixed-width ones included, whose
/// dataset charge does not grow with their pages. Only the decompression
/// and decoding buffers are freed once the chunk is decoded.
fn chunk_transient(
    read: &ChunkPages,
    leaf: Option<&DataType>,
    decode: LeafDecode,
    compressed: bool,
    nested: bool,
    declared: u64,
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
        // polars builds no dictionary for a Boolean column, but still
        // reserves a byte per declared entry when it reads the dictionary
        // page (`read_dict_page`).
        Some(DataType::Boolean) => (0, false),
        Some(other) => (leaf_value_bytes(other, 1), false),
    };
    let dictionary = entry
        .saturating_add(1)
        .saturating_mul(read.dict_entries)
        .saturating_add(if text { read.dict_bytes } else { 0 });
    let entries = declared.max(read.entries);
    let categorical = matches!(
        leaf,
        Some(DataType::Categorical(_, _) | DataType::Enum(_, _))
    );
    let keys = if categorical {
        decode.value_bytes().saturating_mul(entries)
    } else {
        0
    };
    let (levels, reserved) = if nested {
        let values = match leaf {
            // Counted as its keys (and views) above.
            _ if categorical => 0,
            Some(leaf) => leaf_value_bytes(leaf, entries),
            None => session::VIEW_BYTES.saturating_mul(entries),
        };
        (
            LEVEL_BYTES.saturating_mul(read.most_page_entries),
            values.saturating_add(entries.div_ceil(8)),
        )
    } else {
        (0, 0)
    };
    decompressed
        .saturating_add(dictionary)
        .saturating_add(keys)
        .saturating_add(levels)
        .saturating_add(reserved)
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

/// The values a data page of `entries` entries, `nulls` of them null by its
/// header (a v2 header counts them, a v1 header does not: 0), `encoding`
/// and `bytes` uncompressed, holds outside a dictionary, at most (see
/// [`ChunkPages::plain_values`]); none when it is dictionary-encoded.
fn plain_values(
    entries: u64,
    nulls: i32,
    encoding: polars_parquet_format::Encoding,
    bytes: u64,
) -> u64 {
    use polars_parquet_format::Encoding;
    if encoding == Encoding::PLAIN_DICTIONARY || encoding == Encoding::RLE_DICTIONARY {
        return 0;
    }
    let values = entries.saturating_sub(u64::try_from(nulls).unwrap_or(0));
    if encoding == Encoding::PLAIN {
        values.min(bytes / 4)
    } else {
        values
    }
}

/// Page headers parsed between two cancellation checkpoints.
const CHECKPOINT_HEADERS: u64 = 1 << 10;

/// Reads the page headers of a parquet file's column chunks, as polars'
/// page reader (`PageReader`) does before it decodes a chunk, without
/// decompressing or decoding anything.
struct PageWalk<'a> {
    file: std::io::BufReader<SourceReader<'a>>,
    file_len: u64,
    /// Headers parsed so far.
    headers: u64,
}

impl<'a> PageWalk<'a> {
    fn new(file: std::io::BufReader<SourceReader<'a>>, file_len: u64) -> Self {
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
            let at = end - left;
            if !self.skip(size, &mut left) {
                return Ok(read);
            }
            read.dict_bytes = bytes;
            read.dict_entries = entries;
            read.dict_at = Some(at);
            read.dict_size = size;
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
            // Entries, nulls the header counts, and encoding.
            let page = match header.type_ {
                PageType::DATA_PAGE => header
                    .data_page_header
                    .as_ref()
                    .map(|h| (h.num_values, 0, h.encoding)),
                PageType::DATA_PAGE_V2 => header
                    .data_page_header_v2
                    .as_ref()
                    .map(|h| (h.num_values, h.num_nulls, h.encoding)),
                _ => None,
            };
            let size = u64::try_from(header.compressed_page_size);
            let bytes = u64::try_from(header.uncompressed_page_size);
            let (Some((entries, nulls, encoding)), Ok(size), Ok(bytes)) = (page, size, bytes)
            else {
                break;
            };
            if !self.skip(size, &mut left) {
                break;
            }
            seen = seen.saturating_add(i64::from(entries));
            let entries = u64::try_from(entries).unwrap_or(0);
            read.plain_values = read
                .plain_values
                .saturating_add(plain_values(entries, nulls, encoding, bytes));
            read.entries = read.entries.saturating_add(entries);
            read.page_bytes = read.page_bytes.saturating_add(bytes);
            read.largest_page = read.largest_page.max(bytes);
            read.most_page_entries = read.most_page_entries.max(entries);
        }
        Ok(read)
    }

    /// Add the labels of the dictionary page that [`PageWalk::chunk`] found
    /// (`read`), in a chunk compressed with `codec`, to `labels`, each hashed
    /// with `hasher`, as far as `budget` allows (see
    /// [`PageWalk::dictionary_page`]). A page that is not read, or does not
    /// hold the labels its header declares, leaves `labels` to count every
    /// page from its size (see [`page_label_bytes`]). `Err` when the call is
    /// cancelled or past its deadline.
    fn dictionary(
        &mut self,
        read: &ChunkPages,
        codec: polars_parquet_format::CompressionCodec,
        hasher: &std::hash::RandomState,
        labels: &mut DictionaryLabels,
        budget: &mut DictionaryBudget,
    ) -> Result<(), String> {
        let Some(at) = read.dict_at else {
            return Ok(());
        };
        labels.pages = labels
            .pages
            .saturating_add(page_label_bytes(read.dict_bytes, read.dict_entries));
        if labels.unread {
            return Ok(());
        }
        p2a_core::resource::checkpoint()
            .map_err(|e| format!("the call stopped while the file was checked ({e})"))?;
        let counted = self
            .dictionary_page(read, at, codec, budget)
            .is_some_and(|page| labels.add(&page, read.dict_entries, hasher, budget));
        if !counted {
            labels.unread = true;
            labels.seen = std::collections::HashSet::new();
        }
        Ok(())
    }

    /// The dictionary page whose bytes start at `at` (`read` holds its
    /// sizes), decompressed with `codec` as polars does, with polars' own
    /// codecs, into a buffer of the size its header declares. `None` when
    /// `budget` does not allow its bytes, or it is not read or does not
    /// decompress.
    fn dictionary_page(
        &mut self,
        read: &ChunkPages,
        at: u64,
        codec: polars_parquet_format::CompressionCodec,
        budget: &mut DictionaryBudget,
    ) -> Option<Vec<u8>> {
        use polars_parquet::parquet::compression::{Compression, DecompressionContext, decompress};
        use std::io::Read;
        let compressed = codec != polars_parquet_format::CompressionCodec::UNCOMPRESSED;
        let cost = read
            .dict_size
            .saturating_add(if compressed { read.dict_bytes } else { 0 });
        let size = usize::try_from(read.dict_size).ok()?;
        let bytes = usize::try_from(read.dict_bytes).ok()?;
        if !budget.take_bytes(cost) || !self.seek_to(at) {
            return None;
        }
        let mut body = vec![0u8; size];
        self.file.read_exact(&mut body).ok()?;
        if !compressed {
            return Some(body);
        }
        let mut page = vec![0u8; bytes];
        let compression = Compression::try_from(codec).ok()?;
        decompress(
            compression,
            &body,
            &mut page,
            &mut DecompressionContext::Unset,
        )
        .ok()?;
        Some(page)
    }
}

/// Bytes of dictionary pages the check before decoding reads and
/// decompresses at most over a file (see [`DictionaryBudget`]).
const DICTIONARY_READ_BYTES: u64 = 256 << 20;
/// Bytes one dictionary page may take to be read: in the file and
/// decompressed.
const DICTIONARY_PAGE_BYTES: u64 = 64 << 20;
/// Distinct labels whose hashes the check keeps at most, over a file.
const DICTIONARY_LABELS: u64 = 1 << 20;

/// What the check before decoding spends on reading a file's dictionary
/// pages: their bytes in the file and decompressed, and the distinct labels
/// it keeps a hash of. Past it, a column's dictionary pages count from
/// their sizes, as they did before they were read (see
/// [`DictionaryLabels`]).
///
/// The check's own memory, which no quota counts since it runs before the
/// file is admitted: one dictionary page at a time, its bytes from the file
/// and its decompressed bytes, at most [`DICTIONARY_PAGE_BYTES`] together
/// and freed before the next page is read; and the hashes of the distinct
/// labels, at most [`DICTIONARY_LABELS`] over the file's columns, eight
/// bytes and a control byte a slot in sets that hold at least half their
/// slots (about 20 MB, at most twice that when the labels spread over many
/// columns). So about 100 MB at most, held only while the check runs. The
/// hosted deployment runs one call at a time, so one check holds it at
/// once, inside the headroom the memory soft limit leaves above the dataset
/// quotas; and an honest dictionary page is mostly far smaller (arrow's
/// writers fall back to plain pages once one passes about a megabyte).
struct DictionaryBudget {
    /// Bytes left to read and decompress.
    bytes: u64,
    /// Bytes one page may take.
    page: u64,
    /// Distinct labels left.
    labels: u64,
}

impl Default for DictionaryBudget {
    fn default() -> Self {
        Self {
            bytes: DICTIONARY_READ_BYTES,
            page: DICTIONARY_PAGE_BYTES,
            labels: DICTIONARY_LABELS,
        }
    }
}

impl DictionaryBudget {
    /// A budget that reads no page.
    fn none() -> Self {
        Self {
            bytes: 0,
            page: 0,
            labels: 0,
        }
    }

    /// Take `bytes` for one page, when the page and what is left allow it.
    fn take_bytes(&mut self, bytes: u64) -> bool {
        if bytes > self.page || bytes > self.bytes {
            return false;
        }
        self.bytes -= bytes;
        true
    }

    /// Take one distinct label, when any are left.
    fn take_label(&mut self) -> bool {
        if self.labels == 0 {
            return false;
        }
        self.labels -= 1;
        true
    }
}

/// The least a label new to a Categorical column's mapping takes there on
/// top of its text, for a value of a page that is not dictionary-encoded,
/// which may or may not be a label of its own. Measured on a release build
/// of the hosted server (PR-03; the measurements are in session.rs): about
/// 130 B over their text for 8-byte labels loaded from parquet, rounded
/// down here. The quotas charge [`session::LABEL_OVERHEAD_BYTES`] and half
/// the text more, so a column whose values are each a label of their own
/// is not bounded above its charge.
const MAPPING_LABEL_BYTES: u64 = 128;

/// The least the quotas charge for the labels of a dictionary page of
/// `bytes` uncompressed bytes declaring `entries` labels, when its labels
/// are distinct and used: the sum of [`session::label_bytes`] over them.
/// Each label takes a 4-byte length and its text in the page, so the text
/// is `bytes - 4 * entries`, and half of each label's text, rounded down,
/// is at least half of it less a half.
fn page_label_bytes(bytes: u64, entries: u64) -> u64 {
    let text = bytes.saturating_sub(entries.saturating_mul(4));
    text.saturating_add(text.saturating_sub(entries) / 2)
        .saturating_add(entries.saturating_mul(session::LABEL_OVERHEAD_BYTES))
}

/// The labels of one Categorical leaf column over a file's row groups, for
/// [`parquet_decoded_bytes`]: what polars puts into the column's mapping
/// while it decodes it, at what the quotas charge for each label once the
/// file is loaded (see [`session::label_bytes`]).
///
/// polars keeps one mapping a column and puts each label in it once, so a
/// label several row groups' dictionaries hold (each group has a dictionary
/// of its own) is counted once. Counting the largest dictionary instead
/// would let a file whose row groups each hold different labels pass with
/// one group's; counting every page refuses a file whose groups repeat long
/// labels although the quotas would hold it once loaded. A dictionary page
/// that is not read counts from its size (see [`page_label_bytes`]). A label
/// that no row uses counts too (polars writes none; arrow's writers can
/// write a factor's unused levels), since only decoding the data pages
/// tells.
///
/// A value of a data page that is not dictionary-encoded is put into the
/// mapping as it is decoded, and nothing short of decoding the page tells
/// whether it repeats another, so each counts [`MAPPING_LABEL_BYTES`] (its
/// text is in the page's bytes, which the column's values count). Writers
/// fall back to such pages when a dictionary grows past their page limit,
/// that is, for columns of many distinct labels (arrow's writers at about a
/// megabyte; R's and Python's arrow can also be told to write none).
#[derive(Default)]
struct DictionaryLabels {
    /// The hashes of the labels counted.
    seen: std::collections::HashSet<u64>,
    /// What the quotas charge for the labels counted.
    bytes: u64,
    /// What every dictionary page added counts from its size, read or not.
    pages: u64,
    /// Set once a page was not read (see [`PageWalk::dictionary`]).
    unread: bool,
    /// The values of the column's data pages that are not
    /// dictionary-encoded, over its chunks (see
    /// [`ChunkPages::plain_values`]).
    plain_values: u64,
}

impl DictionaryLabels {
    /// Add the `entries` labels of the decompressed dictionary page `page`
    /// (each a 4-byte little-endian length and its bytes, as parquet stores
    /// a dictionary of byte arrays), each hashed with `hasher`, taking each
    /// new one from `budget`. `false` when the page holds fewer labels than
    /// that or the budget runs out.
    fn add(
        &mut self,
        page: &[u8],
        entries: u64,
        hasher: &std::hash::RandomState,
        budget: &mut DictionaryBudget,
    ) -> bool {
        use std::hash::BuildHasher;
        let mut rest = page;
        // Each label takes at least four bytes of the page, so a header
        // that declares more entries than that stops at the page's end.
        for _ in 0..entries {
            let Some((len, tail)) = rest.split_first_chunk::<4>() else {
                return false;
            };
            let Some((label, tail)) = usize::try_from(u32::from_le_bytes(*len))
                .ok()
                .and_then(|len| tail.split_at_checked(len))
            else {
                return false;
            };
            rest = tail;
            if self.seen.insert(hasher.hash_one(label)) {
                if !budget.take_label() {
                    return false;
                }
                self.bytes = self.bytes.saturating_add(session::label_bytes(label.len()));
            }
        }
        true
    }

    /// What the column's labels count: each distinct label once, or every
    /// page from its size when a page was not read; and each value of its
    /// pages that are not dictionary-encoded.
    fn bytes(&self) -> u64 {
        let labels = if self.unread { self.pages } else { self.bytes };
        labels.saturating_add(MAPPING_LABEL_BYTES.saturating_mul(self.plain_values))
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
    /// Below an array of size zero: no values, whatever the entries. The
    /// room polars reserves for the entries such a chunk declares is
    /// counted with the decoder's buffers (see [`chunk_transient`]), so
    /// nothing is lost by counting none here.
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
/// A text leaf takes at least its pages (long strings): the quotas charge
/// more for each value than its pages hold. A Categorical or Enum leaf
/// takes at least its data pages without their headers (the text of a
/// page that is not dictionary-encoded); a Categorical leaf's labels are
/// left to the caller, which counts each label once over the row groups
/// (see [`DictionaryLabels`]), and `holds_labels` gets one entry per leaf
/// consumed, in order: whether it is a Categorical leaf that holds values
/// (an Enum leaf's labels are its type's, see [`parquet_decoded_bytes`]).
/// A fixed-width leaf's charge does not grow with its pages, which also
/// hold levels its values in memory do not; what decoding its pages
/// allocates is counted with the decoder's buffers instead (see
/// [`chunk_transient`]). Types without a fixed floor count nothing.
fn least_value_bytes(
    dtype: &DataType,
    slots: Slots,
    leaves: &mut impl Iterator<Item = LeafChunks>,
    holds_labels: &mut Vec<bool>,
) -> u64 {
    match dtype {
        DataType::List(inner) => match slots {
            // An outermost list in no slots holds nothing but what its leaf
            // columns declare.
            Slots::Known(0) | Slots::Empty => least_value_bytes(inner, slots, leaves, holds_labels),
            Slots::Known(n) => OFFSET_BYTES
                .saturating_mul(n)
                .saturating_add(least_value_bytes(
                    inner,
                    Slots::BelowList {
                        outer: n,
                        nested: false,
                    },
                    leaves,
                    holds_labels,
                )),
            Slots::BelowList { outer, .. } => least_value_bytes(
                inner,
                Slots::BelowList {
                    outer,
                    nested: true,
                },
                leaves,
                holds_labels,
            ),
        },
        DataType::Array(inner, size) => match slots {
            // An array of size zero holds no values; an empty or null one is
            // an entry without them.
            _ if *size == 0 => least_value_bytes(inner, Slots::Empty, leaves, holds_labels),
            Slots::Known(n) => least_value_bytes(
                inner,
                Slots::Known(n.saturating_mul(*size as u64)),
                leaves,
                holds_labels,
            ),
            Slots::Empty | Slots::BelowList { .. } => {
                least_value_bytes(inner, slots, leaves, holds_labels)
            }
        },
        DataType::Struct(fields) => {
            // Every field consumes its own leaves, whichever bound is kept.
            let bounds = fields
                .iter()
                .map(|field| least_value_bytes(field.dtype(), slots, leaves, holds_labels));
            match slots {
                Slots::BelowList { nested: true, .. } => bounds.fold(0u64, u64::max),
                _ => bounds.fold(0u64, u64::saturating_add),
            }
        }
        leaf => {
            let chunks = leaves.next().unwrap_or_default();
            let coded = matches!(leaf, DataType::Categorical(_, _) | DataType::Enum(_, _));
            let values = match slots {
                Slots::Empty => {
                    holds_labels.push(false);
                    return 0;
                }
                Slots::Known(n) => leaf_value_bytes(leaf, n.max(chunks.entries)),
                Slots::BelowList { outer, .. } => {
                    let values = chunks.entries.saturating_sub(outer);
                    leaf_value_bytes(leaf, values).min(OFFSET_BYTES.saturating_mul(values))
                }
            };
            holds_labels.push(matches!(leaf, DataType::Categorical(_, _)));
            match leaf {
                DataType::String | DataType::Binary | DataType::BinaryOffset => {
                    values.max(chunks.page_bytes)
                }
                _ if coded => values.max(chunks.data_page_bytes),
                _ => values,
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
        DataType::Int128 | DataType::UInt128 | DataType::Decimal(_, _) => width(16),
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
/// [`admit_parquet_source`]). Other formats pass: a CSV, Excel, Stata or
/// SAS file is charged only once decoded, and a CSV can decode to several
/// times its size on disk.
async fn admit_parquet(path: &Path, name: &str) -> Result<(), NotAdmitted> {
    let is_parquet = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_lowercase().as_str(), "parquet" | "pq"));
    if !is_parquet {
        return Ok(());
    }
    admit_parquet_source(ParquetSource::File(path), name).await
}

/// Refuse, before decoding it, the parquet file `source` that the dataset
/// quotas would refuse once it was loaded as `name` (see
/// [`session::admit_dataset`]), counting what decoding it allocates on top
/// of the dataset (see [`parquet_decoded_bytes`]).
///
/// A parquet file whose footer declares what no writer produces (see
/// [`read_footer`]) is not loaded whether or not the call has quotas: those
/// checks read the footer alone, and polars' reader does not survive one
/// of them (a negative count reaches an unchecked conversion before a
/// reservation). The page headers and dictionaries are read only under
/// quotas, and a call without them passes after the footer. A file whose
/// footer cannot be read passes (the loader reports its error); one the
/// check will not read is not loaded, and neither is one whose footer or
/// schema makes the check panic: polars' schema conversion
/// (`DataType::from_arrow_field`) panics on some malformed categorical
/// metadata, and the loader runs the same code.
async fn admit_parquet_source(source: ParquetSource<'_>, name: &str) -> Result<(), NotAdmitted> {
    let footer = match std::panic::catch_unwind(|| read_footer(source)) {
        Ok(Ok(Some(footer))) => footer,
        Ok(Ok(None)) => return Ok(()),
        Ok(Err(reason)) => return Err(NotAdmitted::Unread(reason)),
        Err(_) => {
            tracing::info!("the parquet check panicked reading a file's footer");
            return Err(NotAdmitted::Unread(
                "could not read the file's footer".to_string(),
            ));
        }
    };
    if !session::dataset_admission_active() {
        return Ok(());
    }
    let bounded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        footer_decoded_bytes(source, footer, &mut DictionaryBudget::default())
    }));
    let bound = match bounded {
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

/// Decode the uploaded parquet file `bytes` where it lies in memory: an
/// upload is never written to disk, so it needs no writable folder and
/// stores nothing, and the loader's error names no file. The core loader
/// reads it as it reads a file from a path, so a schema it refuses is
/// refused here too, with or without the quotas.
fn load_parquet_upload(bytes: &[u8]) -> Result<Dataset, String> {
    DataLoader::load_parquet_bytes(bytes)
        .map(Dataset::new)
        .map_err(|e| e.to_string())
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
                // Checked and loaded from the request's bytes in memory
                // (see load_parquet_upload): nothing is written to disk.
                // Refused before decoding when the quotas would refuse the
                // decoded dataset.
                match admit_parquet_source(ParquetSource::Bytes(&decoded), &id).await {
                    Ok(()) => {}
                    Err(NotAdmitted::Quota(refusal)) => return Err(refusal),
                    Err(NotAdmitted::Unread(reason)) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse Parquet: {reason}. Nothing was loaded."
                        ))]));
                    }
                }

                match load_parquet_upload(&decoded) {
                    Ok(ds) => ds,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Failed to parse Parquet: {e}"
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
            data_page_bytes: page_bytes,
        });
        least_value_bytes(dtype, Slots::Known(rows), &mut leaves, &mut Vec::new())
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
        // Flat columns count their rows, as before; 128-bit ones 16 bytes a
        // row (UInt128 counted nothing pre-fix).
        assert_eq!(least(&DataType::Float64, 7, &[7]), 56);
        assert_eq!(least(&DataType::UInt128, 3, &[3]), 48);
        assert_eq!(least(&DataType::Int128, 3, &[3]), 48);
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
    fn a_categorical_leaf_leaves_its_dictionary_to_the_caller() {
        // Its codes, or its data pages where they are more; its dictionary
        // page is counted once over the row groups by the caller, which
        // `holds_labels` tells which leaves hold labels.
        use p2a_core::polars::prelude::Categories;
        let cat =
            DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        let pair = fields(&[
            ("k", cat.clone()),
            ("x", DataType::Float64),
            ("e", DataType::Array(Box::new(cat.clone()), 0)),
        ]);
        let chunk = |entries, page_bytes, data_page_bytes| LeafChunks {
            entries,
            page_bytes,
            data_page_bytes,
        };
        let mut leaves = [chunk(10, 5000, 100), chunk(10, 80, 80), chunk(10, 900, 100)].into_iter();
        let mut holds_labels = Vec::new();
        let bytes = least_value_bytes(&pair, Slots::Known(10), &mut leaves, &mut holds_labels);
        assert_eq!(bytes, 100 + 80);
        assert_eq!(holds_labels, [true, false, false]);
        let mut leaves = std::iter::once(chunk(10, 5000, 4000));
        assert_eq!(
            least_value_bytes(&cat, Slots::Known(10), &mut leaves, &mut Vec::new()),
            4000
        );
        // The data pages count without their headers, which can hold the
        // page's least and greatest label and which the chunk's size in the
        // footer counts. That size less the dictionary page, counted before,
        // put a column of long labels above its charge once its labels
        // counted what the quotas charge for them.
        let mut leaves = std::iter::once(chunk(10, 5000, 30));
        assert_eq!(
            least_value_bytes(&cat, Slots::Known(10), &mut leaves, &mut Vec::new()),
            40
        );
        // An Enum leaf counts its codes and data pages the same way, but its
        // labels are its type's categories, which the caller counts once a
        // leaf: its dictionary pages put nothing into a mapping.
        let frozen = p2a_core::polars::prelude::FrozenCategories::new(["lo", "hi"]).unwrap();
        let levels = DataType::from_frozen_categories(frozen);
        let mut leaves = std::iter::once(chunk(10, 5000, 4000));
        let mut holds_labels = Vec::new();
        let bytes = least_value_bytes(&levels, Slots::Known(10), &mut leaves, &mut holds_labels);
        assert_eq!((bytes, holds_labels), (4000, vec![false]));
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
        assert_eq!(
            chunk_transient(&pages, f64, LeafDecode::default(), true, false, 2),
            1000
        );
        // Uncompressed pages are decoded where they lie in the file.
        assert_eq!(
            chunk_transient(&pages, f64, LeafDecode::default(), false, false, 2),
            0
        );
        // A dictionary page: its decompression buffer, and its entries
        // decoded at their width (and a byte each).
        let dict = ChunkPages {
            dict_bytes: 5000,
            dict_entries: 600,
            ..pages
        };
        assert_eq!(
            chunk_transient(&dict, f64, LeafDecode::default(), true, false, 2),
            1000 + 5000 + 9 * 600
        );
        assert_eq!(
            chunk_transient(&dict, f64, LeafDecode::default(), false, false, 2),
            9 * 600
        );
        // A text dictionary is decoded into views, its bytes copied out.
        let text = Some(&DataType::String);
        assert_eq!(
            chunk_transient(&dict, text, LeafDecode::default(), false, false, 2),
            17 * 600 + 5000
        );
        assert_eq!(
            chunk_transient(&dict, None, LeafDecode::default(), false, false, 2),
            17 * 600 + 5000
        );
        // A Boolean dictionary is not decoded, but reading it reserves a
        // byte per entry.
        let flag = Some(&DataType::Boolean);
        assert_eq!(
            chunk_transient(&dict, flag, LeafDecode::default(), false, false, 2),
            600
        );
        // A nested column's levels, two u16 per entry of its largest page,
        // and the room reserved for its entries: two bitmaps for booleans.
        let levels = ChunkPages {
            entries: 1 << 20,
            most_page_entries: 1 << 20,
            ..ChunkPages::default()
        };
        assert_eq!(
            chunk_transient(&levels, flag, LeafDecode::default(), false, true, 1 << 20),
            (4 << 20) + (2 << 17)
        );
        assert_eq!(
            chunk_transient(&levels, flag, LeafDecode::default(), false, false, 1 << 20),
            0
        );
        // The room is reserved for the footer's count before the pages are
        // read, and grows to the pages' entries: the more of the two. Pre-fix
        // neither counted here.
        let none = ChunkPages::default();
        assert_eq!(
            chunk_transient(&none, f64, LeafDecode::default(), false, true, 1 << 30),
            (8 << 30) + (1 << 27)
        );
        assert_eq!(
            chunk_transient(&levels, f64, LeafDecode::default(), false, true, 0),
            (4 << 20) + (8 << 20) + (1 << 17)
        );
        assert_eq!(
            chunk_transient(&none, None, LeafDecode::default(), false, true, 1 << 10),
            16 * 1024 + 128
        );
        // A flat column's reservation is bounded by its rows.
        assert_eq!(
            chunk_transient(&none, f64, LeafDecode::default(), false, false, 1 << 30),
            0
        );
        // A 128-bit leaf reserves 16 bytes an entry (pre-fix nothing).
        let u128 = Some(&DataType::UInt128);
        assert_eq!(
            chunk_transient(&none, u128, LeafDecode::default(), false, true, 1 << 10),
            16 * 1024 + 128
        );
        // A Categorical or Enum leaf's values are decoded into an arrow
        // dictionary first: a key per entry, and a view per entry before it
        // when polars did not write the column. Pre-fix neither counted, so
        // a file of one label over 40,000,000 rows was admitted at its
        // codes' 4 bytes a row.
        use p2a_core::polars::prelude::Categories;
        let cat =
            DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        let cat = Some(&cat);
        let rows = ChunkPages {
            entries: 1000,
            most_page_entries: 1000,
            ..ChunkPages::default()
        };
        let own = LeafDecode {
            key_bytes: 4,
            views: false,
        };
        let other = LeafDecode {
            key_bytes: 4,
            views: true,
        };
        assert_eq!(chunk_transient(&rows, cat, own, false, false, 1000), 4000);
        assert_eq!(
            chunk_transient(&rows, cat, other, false, false, 1000),
            20_000
        );
        // The larger of the footer's and the pages' entries; a key's width
        // is the field's key type's.
        let narrow = LeafDecode {
            key_bytes: 1,
            views: false,
        };
        assert_eq!(
            chunk_transient(&rows, cat, narrow, false, false, 3000),
            3000
        );
        // Nested, the keys (and views) are the decoded chunk, in place of
        // the reservation at the codes' width; levels and validity as for
        // any nested leaf.
        assert_eq!(
            chunk_transient(&rows, cat, other, false, true, 1000),
            20_000 + 4 * 1000 + 125
        );
        // Other leaf types decode no keys.
        assert_eq!(chunk_transient(&rows, f64, other, false, false, 1000), 0);
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
        // 40 labels of about 1,000 bytes, each in every row group: polars
        // writes them into each group's dictionary page, and the loaded
        // column holds each once in its mapping.
        let labels = Series::new(
            "labels".into(),
            (0..rows)
                .map(|i| format!("{:01000}", (i * 7) % 40))
                .collect::<Vec<_>>(),
        )
        .cast(&cat)
        .unwrap();
        // 2,000 distinct 6-byte labels, in a mapping of their own: short
        // labels, whose charge is mostly the mapping's overhead.
        let short_cat = DataType::from_categories(p2a_core::polars::prelude::Categories::random(
            "".into(),
            CategoricalPhysical::U32,
        ));
        let short = Series::new(
            "short".into(),
            (0..rows).map(|i| format!("s{i:05}")).collect::<Vec<_>>(),
        )
        .cast(&short_cat)
        .unwrap();
        // An Enum of 50 categories of 200 bytes, 30 of them used: the column
        // owns all 50.
        let names: Vec<String> = (0..50).map(|i| format!("{i:0200}")).collect();
        let frozen =
            p2a_core::polars::prelude::FrozenCategories::new(names.iter().map(String::as_str))
                .unwrap();
        let levels = Series::new(
            "levels".into(),
            (0..rows)
                .map(|i| names[(i * 7) % 30].clone())
                .collect::<Vec<_>>(),
        )
        .cast(&DataType::from_frozen_categories(frozen.clone()))
        .unwrap();
        let mut df = DataFrame::new(vec![
            lists.into_series().with_name("lists".into()).into(),
            nested.into_series().with_name("nested".into()).into(),
            structs.into_series().with_name("structs".into()).into(),
            cats.into_series().with_name("cats".into()).into(),
            deep.into_series().with_name("deep".into()).into(),
            array.into(),
            labels.into(),
            short.into(),
            levels.into(),
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
        // In one row group and in several: the bound is summed over the row
        // groups, a categorical's labels excepted. Pre-fix the labels
        // column's bound in several groups counted its dictionary once per
        // group, more than three times its charge.
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
            for name in [
                "lists", "nested", "structs", "cats", "deep", "arr", "labels", "short", "levels",
            ] {
                let mut one = df.select([name]).unwrap();
                let (charge, bound) = loaded_charge_and_bound(&mut one, groups);
                assert!(
                    bound <= charge,
                    "{groups:?} {name}: bound {bound} > charge {charge}"
                );
                // A Categorical column's labels count once, at what the
                // quotas charge for them (read from the pages over several
                // row groups, from the page's size in one); an Enum's
                // categories once. Pre-fix a label counted its bytes in the
                // page (1,004 and 10 bytes) and an Enum's categories nothing.
                let least = match name {
                    "labels" => 40 * session::label_bytes(1_000),
                    "short" if groups.is_some() => 2_000 * session::label_bytes(6),
                    "short" => 2_000 * 168,
                    "levels" => session::frozen_label_bytes(&frozen),
                    _ => 0,
                };
                assert!(bound >= least, "{groups:?} {name}: bound {bound} < {least}");
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
    fn a_chunk_that_declares_a_negative_count_is_not_read() {
        // polars converts a column chunk's footer count to usize as it is
        // (`PageReader::total_num_values`) and, for a nested column,
        // reserves room for that many values before it decodes the chunk:
        // for a struct of booleans, two bitmaps of nearly 2^64 bits, a
        // request the allocator cannot meet, which aborts the process.
        // Pre-fix the walk counted such a chunk as holding no entries.
        use p2a_core::polars::prelude::{IntoSeries, NamedFrom, Series, StructChunked};
        let flags = Series::new("b".into(), [true, false, true]);
        let s = StructChunked::from_series("s".into(), 3, [flags].iter())
            .unwrap()
            .into_series();
        let mut df = DataFrame::new(vec![s.into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("negative.parquet");
        write_parquet(&mut df, &path, 3);
        assert!(parquet_decoded_bytes(&path).unwrap().is_some());
        rewrite_footer(&path, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                chunk.meta_data.as_mut().unwrap().num_values = -1;
            }
        });
        let reason = parquet_decoded_bytes(&path).unwrap_err();
        assert_eq!(
            reason,
            "a column chunk of the file declares a negative number of values"
        );
    }

    /// A parquet file of `rows` rows at `path` whose column `a` is an array
    /// of size zero of floats.
    fn empty_arrays_parquet(path: &Path, rows: usize) {
        use p2a_core::polars::prelude::{IntoSeries, ListChunked, Series};
        let lists: ListChunked = (0..rows)
            .map(|_| Some(Series::new_empty("".into(), &DataType::Float64)))
            .collect();
        let arrays = lists
            .into_series()
            .cast(&DataType::Array(Box::new(DataType::Float64), 0))
            .unwrap()
            .with_name("a".into());
        let mut df = DataFrame::new(vec![arrays.into()]).unwrap();
        write_parquet(&mut df, path, rows);
    }

    #[test]
    fn a_nested_chunk_reserves_the_entries_its_footer_declares() {
        // An array of size zero holds no values, whatever its chunk says,
        // but polars still reserves room for the chunk's footer count at
        // the leaf's width (and a validity bit each) before it decodes it
        // (`collect_nested`). Pre-fix nothing counted that reservation, so
        // a small file whose chunk declares billions of entries passed.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.parquet");
        empty_arrays_parquet(&path, 1_000);
        // An honest file of them is small before and after.
        let honest = bound_of(&path);
        assert!(honest.dataset + honest.transient < 1 << 20, "{honest:?}");
        let loaded = DataLoader::load(&path).unwrap();
        assert_eq!(
            loaded.df().dtypes(),
            [DataType::Array(Box::new(DataType::Float64), 0)]
        );
        let charge = session::datasets_bytes_of([("d", &loaded)]);
        assert!(honest.dataset <= charge, "{honest:?} > {charge}");
        rewrite_footer(&path, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                chunk.meta_data.as_mut().unwrap().num_values = 1 << 31;
            }
        });
        let bound = bound_of(&path);
        assert!(bound.transient >= 8 << 31, "{bound:?}");
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
    fn walk_of(path: &Path) -> PageWalk<'_> {
        let (file, file_len) = ParquetSource::File(path).open().unwrap();
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
        rewrite_footer(&path, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                let meta = chunk.meta_data.as_mut().unwrap();
                meta.num_values = 0;
                meta.total_uncompressed_size = 1;
            }
        });
        let bound = bound_of(&path);
        assert!(bound.dataset >= 2_000 * 1_000, "{bound:?}");
        // Its decoded labels and their views, too.
        assert!(bound.transient >= 2_000 * (1_000 + 16), "{bound:?}");
        // A negative count is not read at all (see
        // a_chunk_that_declares_a_negative_count_is_not_read).
        rewrite_footer(&path, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                chunk.meta_data.as_mut().unwrap().num_values = -5;
            }
        });
        assert!(parquet_decoded_bytes(&path).is_err());
    }

    /// A parquet file at `path` of 2,000 rows in row groups of 300 whose
    /// column `k` is a Categorical of about 1,000-byte labels: `distinct` of
    /// them, the row `i` holding label `i % distinct`.
    fn labels_parquet(path: &Path, distinct: usize) {
        use p2a_core::polars::prelude::{Categories, NamedFrom, Series};
        let values: Vec<String> = (0..2_000)
            .map(|i| format!("k{:0999}", i % distinct))
            .collect();
        let dtype =
            DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        let k = Series::new("k".into(), values).cast(&dtype).unwrap();
        let mut df = DataFrame::new(vec![k.into()]).unwrap();
        write_parquet(&mut df, path, 300);
    }

    #[test]
    fn a_label_counts_once_however_many_row_groups_hold_it() {
        // Several row groups. Labels repeated in each group's dictionary
        // count once, as the loaded column's mapping holds them; labels that
        // each group holds alone all count, as the mapping holds them all.
        // The largest dictionary alone would let the second file pass with
        // one group's few hundred labels.
        let dir = tempfile::tempdir().unwrap();
        let repeated = dir.path().join("repeated.parquet");
        labels_parquet(&repeated, 40);
        let groups = ParquetReader::new(std::fs::File::open(&repeated).unwrap())
            .get_metadata()
            .unwrap()
            .row_groups
            .len() as u64;
        assert!(groups >= 3, "{groups}");
        let label = session::label_bytes(1_000);
        let bound = bound_of(&repeated).dataset;
        assert!(bound >= 40 * label, "{bound}");
        assert!(bound < 2 * 40 * label + 2_000 * 4 + 8_192, "{bound}");
        let distinct = dir.path().join("distinct.parquet");
        labels_parquet(&distinct, 2_000);
        let bound = bound_of(&distinct).dataset;
        assert!(bound >= 2_000 * label, "{bound}");
        // Past the budget the pages count from their sizes: more than the
        // labels once, never less.
        let mut tight = DictionaryBudget {
            bytes: 100_000,
            ..DictionaryBudget::default()
        };
        let bounded = parquet_decoded_bytes_within(ParquetSource::File(&repeated), &mut tight)
            .unwrap()
            .unwrap()
            .dataset;
        assert!(bounded >= groups * 40 * (label - 1), "{groups}: {bounded}");
        let mut few = DictionaryBudget {
            labels: 10,
            ..DictionaryBudget::default()
        };
        let bounded = parquet_decoded_bytes_within(ParquetSource::File(&distinct), &mut few)
            .unwrap()
            .unwrap()
            .dataset;
        assert!(bounded >= 2_000 * (label - 1), "{bounded}");
    }

    #[test]
    fn short_labels_count_what_the_quotas_charge_for_them() {
        // 20,000 distinct 6-byte labels over seven row groups. polars puts
        // each into the column's mapping at about 130 bytes over its text,
        // and the quotas charge 169 bytes for it. Pre-fix the bound counted
        // its 10 bytes in the page, so a file of millions of short labels
        // was admitted far below what loading it took.
        use p2a_core::polars::prelude::{Categories, NamedFrom, Series};
        let rows = 20_000usize;
        let dtype =
            DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        let k = Series::new(
            "k".into(),
            (0..rows).map(|i| format!("s{i:05}")).collect::<Vec<_>>(),
        )
        .cast(&dtype)
        .unwrap();
        let mut df = DataFrame::new(vec![k.into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("short.parquet");
        write_parquet(&mut df, &path, 3_000);
        let bound = bound_of(&path).dataset;
        assert!(bound >= rows as u64 * session::label_bytes(6), "{bound}");
        let charge = session::datasets_bytes_of([("d", &DataLoader::load(&path).unwrap())]);
        assert!(bound <= charge, "{bound} > {charge}");
    }

    /// Rewrite the header of the first page of the first column chunk of the
    /// parquet file at `path` with `edit`, which must leave its encoded size
    /// as it was.
    fn rewrite_first_page_header(
        path: &Path,
        edit: impl FnOnce(&mut polars_parquet_format::PageHeader),
    ) {
        use polars_parquet_format::thrift::protocol::{
            TCompactInputProtocol, TCompactOutputProtocol,
        };
        let metadata = std::sync::Arc::clone(
            ParquetReader::new(std::fs::File::open(path).unwrap())
                .get_metadata()
                .unwrap(),
        );
        let at = metadata.row_groups[0].parquet_columns()[0]
            .byte_range()
            .start as usize;
        let mut bytes = std::fs::read(path).unwrap();
        let mut rest = &bytes[at..];
        let mut header = polars_parquet_format::PageHeader::read_from_in_protocol(
            &mut TCompactInputProtocol::new(&mut rest, usize::MAX),
        )
        .unwrap();
        let len = bytes.len() - at - rest.len();
        edit(&mut header);
        let mut written = Vec::new();
        header
            .write_to_out_protocol(&mut TCompactOutputProtocol::new(&mut written))
            .unwrap();
        assert_eq!(written.len(), len);
        bytes[at..at + len].copy_from_slice(&written);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn an_unreadable_dictionary_page_counts_every_page_from_its_size() {
        // Seven row groups, each with a dictionary page of the same 40
        // labels of 1,000 bytes; the first group's header is rewritten to
        // declare 63 labels, more than its page holds. Its labels are not
        // read, so the column counts every page from its size, whatever the
        // others hold: at least each page's bytes.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("unread.parquet");
        labels_parquet(&path, 40);
        rewrite_first_page_header(&path, |header| {
            let dict = header.dictionary_page_header.as_mut().unwrap();
            assert_eq!(dict.num_values, 40);
            dict.num_values = 63;
        });
        let metadata = std::sync::Arc::clone(
            ParquetReader::new(std::fs::File::open(&path).unwrap())
                .get_metadata()
                .unwrap(),
        );
        assert!(metadata.row_groups.len() >= 2);
        let mut walk = walk_of(&path);
        let mut pages = 0;
        for group in &metadata.row_groups {
            let chunk = &group.parquet_columns()[0];
            let read = walk.chunk(chunk.byte_range(), chunk.num_values()).unwrap();
            assert!(read.dict_bytes >= 40 * 1_004);
            pages += read.dict_bytes;
        }
        let bound = bound_of(&path).dataset;
        assert!(bound >= pages, "{bound} < {pages}");
        // Read, the 40 labels counted once: far less.
        let honest = dir.path().join("honest.parquet");
        labels_parquet(&honest, 40);
        assert!(bound_of(&honest).dataset < pages, "{pages}");
    }

    #[test]
    fn a_dictionary_page_that_does_not_hold_its_labels_is_not_counted_from_them() {
        // A page whose header declares more labels than it holds, or a
        // label longer than the page, is not counted from its labels. A
        // label it holds counts what the quotas charge for it, once.
        let hasher = std::hash::RandomState::new();
        let mut page = Vec::new();
        for label in [b"ab".as_slice(), b"cde", b"ab"] {
            page.extend_from_slice(&(label.len() as u32).to_le_bytes());
            page.extend_from_slice(label);
        }
        let charged = session::label_bytes(2) + session::label_bytes(3);
        assert_eq!(charged, 163 + 164);
        let mut labels = DictionaryLabels::default();
        let mut budget = DictionaryBudget::default();
        assert!(labels.add(&page, 3, &hasher, &mut budget));
        assert_eq!(labels.bytes(), charged);
        assert!(labels.add(&page, 3, &hasher, &mut budget));
        assert_eq!(labels.bytes(), charged);
        assert!(!labels.add(&page, 4, &hasher, &mut budget));
        let mut long = 100u32.to_le_bytes().to_vec();
        long.extend_from_slice(b"short");
        assert!(!DictionaryLabels::default().add(&long, 1, &hasher, &mut budget));
        // Each entry takes at least four bytes, so a header that declares
        // billions of entries stops at the page's end.
        assert!(!DictionaryLabels::default().add(&page, u64::MAX, &hasher, &mut budget));
        // A value of a page that is not dictionary-encoded counts the
        // mapping's least overhead for a label, whether or not it repeats.
        let mut plain = DictionaryLabels {
            plain_values: 10,
            ..DictionaryLabels::default()
        };
        assert_eq!(plain.bytes(), 10 * MAPPING_LABEL_BYTES);
        assert!(plain.add(&page, 3, &hasher, &mut budget));
        assert_eq!(plain.bytes(), charged + 10 * MAPPING_LABEL_BYTES);
        // Not read, every page counts from its size, plus those values.
        plain.unread = true;
        plain.pages = page_label_bytes(page.len() as u64, 3);
        assert_eq!(plain.bytes(), plain.pages + 10 * MAPPING_LABEL_BYTES);
    }

    #[test]
    fn a_dictionary_pages_size_bounds_what_its_labels_are_charged() {
        // Its labels' text, and half of it less a half a label, and the
        // mapping's overhead for each: at most what the quotas charge for
        // the page's labels when they are distinct and used, and at least
        // the page's bytes. Pre-fix it was the page's bytes, 4 + 1,000 for a
        // 1,000-byte label, against a charge of 1,660.
        let page = |lens: &[usize]| lens.iter().map(|len| 4 + *len as u64).sum::<u64>();
        let charge = |lens: &[usize]| lens.iter().map(|len| session::label_bytes(*len)).sum();
        for lens in [&[2usize, 3][..], &[2, 2], &[1_000; 40], &[0, 7, 13, 1]] {
            let bytes = page(lens);
            let entries = lens.len() as u64;
            let counted = page_label_bytes(bytes, entries);
            assert!(counted <= charge(lens), "{lens:?}: {counted}");
            assert!(counted >= bytes, "{lens:?}: {counted}");
            assert!(
                counted + entries / 2 + 1 >= charge(lens),
                "{lens:?}: {counted}"
            );
        }
        assert_eq!(page_label_bytes(40 * 1_004, 40), 40_000 + 19_980 + 40 * 160);
        // A header that declares more labels than the page holds bytes for
        // counts the overhead of each.
        assert_eq!(page_label_bytes(10, 1_000), 160_000);
    }

    #[test]
    fn values_outside_a_dictionary_are_counted_from_the_page_headers() {
        use polars_parquet_format::Encoding;
        // A dictionary-encoded page holds labels of the dictionary only.
        assert_eq!(plain_values(1_000, 0, Encoding::RLE_DICTIONARY, 9_000), 0);
        assert_eq!(plain_values(1_000, 0, Encoding::PLAIN_DICTIONARY, 9_000), 0);
        // A PLAIN page holds no more values than a quarter of its bytes (a
        // value takes a 4-byte length and its text), and no more than its
        // entries less the nulls a v2 header counts.
        assert_eq!(plain_values(1_000, 0, Encoding::PLAIN, 2_000), 500);
        assert_eq!(plain_values(1_000, 0, Encoding::PLAIN, 12_000), 1_000);
        assert_eq!(plain_values(1_000, 300, Encoding::PLAIN, 12_000), 700);
        assert_eq!(plain_values(1_000, -5, Encoding::PLAIN, 12_000), 1_000);
        // Other encodings can pack values tighter than their bytes.
        assert_eq!(
            plain_values(1_000, 0, Encoding::DELTA_BYTE_ARRAY, 10),
            1_000
        );
    }

    /// Rewrite the ARROW:schema of the parquet file at `path` so that no
    /// field carries polars' categorical or enum metadata (see
    /// [`POLARS_DICTIONARY_KEYS`]), as a file of another writer has none:
    /// each key is overwritten in place by a key of the same length.
    fn strip_polars_metadata(path: &Path) {
        use base64::Engine;
        rewrite_footer(path, |footer| {
            let schema = footer
                .key_value_metadata
                .as_mut()
                .unwrap()
                .iter_mut()
                .find(|kv| kv.key == "ARROW:schema")
                .unwrap();
            let engine = base64::engine::general_purpose::STANDARD;
            let mut ipc = engine.decode(schema.value.as_ref().unwrap()).unwrap();
            let mut stripped = 0;
            for at in 0..ipc.len() {
                if ipc[at..].starts_with(b"_PL_") {
                    ipc[at..at + 4].copy_from_slice(b"_XX_");
                    stripped += 1;
                }
            }
            assert!(stripped > 0);
            schema.value = Some(engine.encode(ipc));
        });
    }

    /// The [`LeafDecode`]s of the parquet file at `path`, its leaves in
    /// order.
    fn decodes_of(path: &Path) -> Vec<LeafDecode> {
        let mut footer = read_footer(ParquetSource::File(path)).unwrap().unwrap();
        let schema = footer.reader.schema().unwrap();
        let mut out = Vec::new();
        for field in schema.iter_values() {
            leaf_decodes(&DataType::from_arrow_field(field), Some(field), &mut out);
        }
        out
    }

    #[test]
    fn a_categorical_columns_values_take_a_key_and_a_view_unless_polars_wrote_it() {
        // polars decodes a dictionary<string> column it wrote (its field
        // metadata says so) straight into keys; any other one into string
        // views first, which it then casts. One label over 100,000 rows in
        // one row group: the views alone are 1.6 MB. Pre-fix neither the
        // keys nor the views counted.
        use p2a_core::polars::prelude::{Categories, IntoSeries, ListChunked, NamedFrom, Series};
        let rows = 100_000usize;
        let cat = |physical| DataType::from_categories(Categories::random("".into(), physical));
        let one = Series::new("one".into(), vec!["a"; rows])
            .cast(&cat(CategoricalPhysical::U32))
            .unwrap();
        let small = Series::new(
            "small".into(),
            (0..rows).map(|i| format!("v{}", i % 3)).collect::<Vec<_>>(),
        )
        .cast(&cat(CategoricalPhysical::U8))
        .unwrap();
        let in_lists: ListChunked = (0..rows).map(|i| Some(one.slice(0, i % 3))).collect();
        let in_lists = in_lists.into_series().with_name("lists".into());
        let flat = Series::new(
            "f".into(),
            (0..rows).map(|i| i as f64 * 0.5 + 0.1).collect::<Vec<_>>(),
        );
        let mut df =
            DataFrame::new(vec![one.into(), small.into(), in_lists.into(), flat.into()]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("own.parquet");
        write_parquet(&mut df, &path, rows);
        let own = |key_bytes| LeafDecode {
            key_bytes,
            views: false,
        };
        assert_eq!(
            decodes_of(&path),
            [own(4), own(1), own(4), LeafDecode::default()]
        );
        let before = bound_of(&path);
        strip_polars_metadata(&path);
        let other = |key_bytes| LeafDecode {
            key_bytes,
            views: true,
        };
        assert_eq!(
            decodes_of(&path),
            [other(4), other(1), other(4), LeafDecode::default()]
        );
        let after = bound_of(&path);
        // Without polars' metadata the 8-bit column loads with polars'
        // default 32-bit codes, three bytes a row more; the columns still
        // load as Categorical, and the bound stays below their charge.
        assert_eq!(after.dataset, before.dataset + 3 * rows as u64);
        let loaded = DataLoader::load(&path).unwrap();
        for name in ["one", "small"] {
            let dtype = loaded.df().column(name).unwrap().dtype().clone();
            assert_eq!(dtype.cat_physical().unwrap(), CategoricalPhysical::U32);
        }
        let charge = session::datasets_bytes_of([("d", &loaded)]);
        assert!(after.dataset <= charge, "{after:?} > {charge}");
        // Each value's view, in the chunks decoded at once.
        let threads = decode_threads() as u64;
        let views = if threads >= 2 { 2 } else { 1 } * rows as u64 * session::VIEW_BYTES;
        assert!(
            after.transient >= before.transient + views,
            "{before:?} {after:?}"
        );
    }

    #[test]
    fn an_upload_decodes_from_its_bytes_and_its_error_names_no_file() {
        // Pre-fix an upload was written to a file in the server's temporary
        // folder and loaded from there. From its bytes it decodes to the
        // frame its file loads to, and the loader's error has no file to
        // name.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.parquet");
        let mut df = df!(
            "x" => [1i64, 2, 3, 4],
            "y" => [0.5, 1.75, 2.25, 3.1],
            "s" => ["a", "bb", "ccc", "dddd"],
        )
        .unwrap();
        write_parquet(&mut df, &path, 2);
        let uploaded = load_parquet_upload(&std::fs::read(&path).unwrap()).unwrap();
        assert!(
            uploaded
                .df()
                .equals_missing(DataLoader::load(&path).unwrap().df())
        );
        let Err(shown) = load_parquet_upload(b"x,y\n1,2\n") else {
            panic!("CSV text decoded as parquet");
        };
        assert!(!shown.contains(".parquet"), "{shown}");
        assert!(
            !shown.contains(&dir.path().display().to_string()),
            "{shown}"
        );
    }

    /// [`parquet_decoded_bytes`] of the parquet file at `path`, read from
    /// its bytes in memory, as an upload's are.
    fn bytes_bound_of(path: &Path) -> Result<Option<DecodedBound>, String> {
        let bytes = std::fs::read(path).unwrap();
        parquet_decoded_bytes_within(
            ParquetSource::Bytes(&bytes),
            &mut DictionaryBudget::default(),
        )
    }

    #[test]
    fn an_uploads_bytes_are_checked_as_its_file_is() {
        // An upload is checked from its bytes, a loaded file from disk: the
        // same file gives the same bound, or the same refusal.
        use p2a_core::polars::prelude::{Categories, NamedFrom, Series};
        let dir = tempfile::tempdir().unwrap();
        // Labels over several row groups: their dictionary pages are read.
        let labels = dir.path().join("labels.parquet");
        let dtype =
            DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        let k = Series::new(
            "k".into(),
            (0..3_000)
                .map(|i| format!("s{:04}", i % 700))
                .collect::<Vec<_>>(),
        )
        .cast(&dtype)
        .unwrap();
        write_parquet(&mut DataFrame::new(vec![k.into()]).unwrap(), &labels, 1_000);
        // A list column: its page headers give its entries.
        let lists = dir.path().join("lists.parquet");
        let inner = Series::new(
            "".into(),
            (0..5_000)
                .map(|i| (i % 13) as f64 + 0.5)
                .collect::<Vec<_>>(),
        );
        let mut df = DataFrame::new(vec![Series::new("l".into(), [inner]).into()]).unwrap();
        write_parquet(&mut df, &lists, 1);
        let [_, honest] = [&labels, &lists].map(|path| {
            let from_file = parquet_decoded_bytes(path).unwrap().unwrap();
            assert!(from_file.dataset > 0, "{from_file:?}");
            assert_eq!(bytes_bound_of(path), Ok(Some(from_file)), "{path:?}");
            from_file
        });
        // A footer that understates the entries: the file's bound still
        // comes from its page headers, and so does the bytes'.
        rewrite_footer(&lists, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                chunk.meta_data.as_mut().unwrap().num_values = 1;
            }
        });
        let understated = parquet_decoded_bytes(&lists).unwrap().unwrap();
        assert_eq!(understated.dataset, honest.dataset, "{understated:?}");
        assert_eq!(bytes_bound_of(&lists), Ok(Some(understated)));
        // A chunk that passes the end of the file, and a negative count.
        rewrite_footer(&labels, |footer| {
            let chunk = &mut footer.row_groups[0].columns[0];
            chunk.meta_data.as_mut().unwrap().total_compressed_size = 1 << 40;
        });
        rewrite_footer(&lists, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                chunk.meta_data.as_mut().unwrap().num_values = -1;
            }
        });
        for path in [&labels, &lists] {
            let from_file = parquet_decoded_bytes(path).unwrap_err();
            assert_eq!(bytes_bound_of(path), Err(from_file), "{path:?}");
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
