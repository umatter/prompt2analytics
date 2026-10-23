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

/// The least the dataset quotas will charge for the parquet file at `path`
/// once it is decoded, read from its footer and page headers before
/// anything is decoded: each column's values at their in-memory width, a
/// list's from the entries counted below it (see [`least_value_bytes`]),
/// plus the dataset's structure. A small file can declare hundreds of
/// millions of rows, or one row holding a list of as many values (one value
/// run-length encoded), and the quotas only see a dataset once it is built.
/// `None` when the footer cannot be read; the loader then reports the
/// file's error.
///
/// The counts are the ones the decoder follows, not the footer's summary of
/// them: the rows are each row group's own (polars decodes those and never
/// reads the file-level count), and a leaf column's entries and page bytes
/// are what the page headers declare over the pages the decoder reads (see
/// [`read_chunk_pages`]) where that is more than the column chunk's
/// footer entry says. So a footer that understates either is still charged
/// what decoding the file allocates.
fn parquet_decoded_bytes(path: &Path) -> Option<u64> {
    let mut reader = ParquetReader::new(std::fs::File::open(path).ok()?);
    let metadata = std::sync::Arc::clone(reader.get_metadata().ok()?);
    let rows = metadata
        .row_groups
        .iter()
        .map(|group| group.num_rows() as u64)
        .fold(0u64, u64::saturating_add);
    // A second handle for the page headers: the reader keeps its own.
    let file = std::fs::File::open(path).ok()?;
    let file_len = file.metadata().ok()?.len();
    let mut pages = std::io::BufReader::new(file);
    // What is known about each leaf column under the top-level column
    // `root`, summed over the row groups, in the order of the file's columns,
    // which is the order of the column's leaves.
    let mut leaves_of = |root: &str| {
        let mut leaves: Vec<LeafChunks> = Vec::new();
        for group in &metadata.row_groups {
            let chunks = group.columns_under_root_iter(root).into_iter().flatten();
            for (i, chunk) in chunks.enumerate() {
                let read =
                    read_chunk_pages(&mut pages, chunk.byte_range(), chunk.num_values(), file_len);
                let entries = u64::try_from(chunk.num_values())
                    .unwrap_or(0)
                    .max(read.entries);
                let page_bytes = u64::try_from(chunk.uncompressed_size())
                    .unwrap_or(0)
                    .max(read.page_bytes);
                if i == leaves.len() {
                    leaves.push(LeafChunks::default());
                }
                let leaf = &mut leaves[i];
                leaf.entries = leaf.entries.saturating_add(entries);
                leaf.page_bytes = leaf.page_bytes.saturating_add(page_bytes);
            }
        }
        leaves
    };
    let schema = reader.schema().ok()?;
    let values = schema.iter_values().fold(0u64, |bytes, field| {
        let mut leaves = leaves_of(field.name.as_str()).into_iter();
        let dtype = DataType::from_arrow_field(field);
        bytes.saturating_add(least_value_bytes(&dtype, Slots::Known(rows), &mut leaves))
    });
    let structure = session::structure_bytes(schema.iter_names().map(|name| name.as_str()));
    Some(values.saturating_add(structure))
}

/// What is known about one leaf column before decoding, summed over the row
/// groups.
#[derive(Clone, Copy, Default)]
struct LeafChunks {
    /// Its entries: values, nulls, and empty or null lists above them.
    entries: u64,
    /// Its pages' uncompressed size.
    page_bytes: u64,
}

/// What the page headers of one column chunk declare.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ChunkPages {
    /// The entries of its data pages.
    entries: u64,
    /// The uncompressed size of its pages, dictionary page included.
    page_bytes: u64,
}

/// Read the page headers of the column chunk at `range` of a parquet file
/// of `file_len` bytes, as polars' page reader (`PageReader`) does before
/// it decodes the chunk, without decompressing or decoding anything: from
/// the chunk's start, a header and then a page at a time, while the data
/// pages' declared entries sum to less than `declared` (the chunk's footer
/// entry, signed as the reader keeps it; the reader counts a whole page at
/// a time, so the last page it reads can hold far more). It stops where the
/// reader fails: a header that does not parse, a page type it does not
/// read, a page without its type's header, or a page past the chunk's end.
/// Headers are parsed with no more than the chunk's bytes, so a header
/// cannot make this allocate more.
fn read_chunk_pages(
    file: &mut std::io::BufReader<std::fs::File>,
    range: std::ops::Range<u64>,
    declared: i64,
    file_len: u64,
) -> ChunkPages {
    use polars_parquet_format::thrift::protocol::TCompactInputProtocol;
    use polars_parquet_format::{PageHeader, PageType};
    use std::io::{Read, Seek, SeekFrom};

    let mut read = ChunkPages::default();
    let end = range.end.min(file_len);
    if range.start >= end || file.seek(SeekFrom::Start(range.start)).is_err() {
        return read;
    }
    let mut left = end - range.start;
    // The reader's own tally, which a negative page count lowers.
    let mut seen = 0i64;
    while seen < declared {
        let mut chunk = (&mut *file).take(left);
        let header = {
            let limit = usize::try_from(left).unwrap_or(usize::MAX);
            let mut input = TCompactInputProtocol::new(&mut chunk, limit);
            PageHeader::read_from_in_protocol(&mut input)
        };
        left = chunk.limit();
        let Ok(header) = header else {
            break;
        };
        let entries = match header.type_ {
            PageType::DATA_PAGE => header.data_page_header.as_ref().map(|h| h.num_values),
            PageType::DATA_PAGE_V2 => header.data_page_header_v2.as_ref().map(|h| h.num_values),
            PageType::DICTIONARY_PAGE => header.dictionary_page_header.as_ref().map(|_| 0),
            _ => None,
        };
        let (Some(entries), Ok(size)) = (entries, u64::try_from(header.compressed_page_size))
        else {
            break;
        };
        if size > left {
            break;
        }
        seen = seen.saturating_add(i64::from(entries));
        read.entries = read
            .entries
            .saturating_add(u64::try_from(entries).unwrap_or(0));
        read.page_bytes = read
            .page_bytes
            .saturating_add(u64::try_from(header.uncompressed_page_size).unwrap_or(0));
        // `size` is at most `left`, which came from two u64 file offsets
        // the footer stores as i64.
        let Ok(skip) = i64::try_from(size) else {
            break;
        };
        if file.seek_relative(skip).is_err() {
            break;
        }
        left -= size;
    }
    read
}

/// How many values a column, or a part of a nested one, holds, for
/// [`least_value_bytes`].
#[derive(Clone, Copy)]
enum Slots {
    /// No list above: the rows, times the sizes of the arrays above.
    Known(u64),
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
/// value and label than its pages hold. Types without a fixed floor count
/// nothing.
fn least_value_bytes(
    dtype: &DataType,
    slots: Slots,
    leaves: &mut impl Iterator<Item = LeafChunks>,
) -> u64 {
    match dtype {
        DataType::List(inner) => match slots {
            // An outermost list in no slots holds nothing.
            Slots::Known(0) => least_value_bytes(inner, slots, leaves),
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
            Slots::Known(n) => {
                least_value_bytes(inner, Slots::Known(n.saturating_mul(*size as u64)), leaves)
            }
            // An empty or null array of size zero is an entry without values.
            Slots::BelowList { .. } if *size == 0 => {
                least_value_bytes(inner, Slots::Known(0), leaves)
            }
            Slots::BelowList { .. } => least_value_bytes(inner, slots, leaves),
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
                Slots::Known(0) => return 0,
                Slots::Known(n) => leaf_value_bytes(leaf, n),
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

/// Refuse, before decoding it, a parquet file at `path` that the dataset
/// quotas would refuse once it was loaded as `name` (see
/// [`session::admit_dataset`]). Other formats, and calls without quotas,
/// pass: a CSV, Excel, Stata or SAS file is charged only once decoded, and a
/// CSV can decode to several times its size on disk.
async fn admit_parquet(path: &Path, name: &str) -> Result<(), McpError> {
    let is_parquet = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| matches!(e.to_lowercase().as_str(), "parquet" | "pq"));
    if !is_parquet || !session::dataset_admission_active() {
        return Ok(());
    }
    let Some(at_least) = parquet_decoded_bytes(path) else {
        return Ok(());
    };
    session::admit_dataset(name, at_least)
        .await
        .map_err(session::quota_refusal)
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
        admit_parquet(&path, &id).await?;

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
                // Write to temp file and load (Parquet is binary format)
                let temp_dir = std::env::temp_dir();
                let temp_path = temp_dir.join(format!("upload_{}.parquet", uuid::Uuid::new_v4()));

                if let Err(e) =
                    std::fs::File::create(&temp_path).and_then(|mut f| f.write_all(&decoded))
                {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Failed to write temp file: {}",
                        e
                    ))]));
                }

                // Refused before decoding when the quotas would refuse the
                // decoded dataset.
                let result = match admit_parquet(&temp_path, &id).await {
                    Ok(()) => DataLoader::load(&temp_path),
                    Err(refusal) => {
                        let _ = std::fs::remove_file(&temp_path);
                        return Err(refusal);
                    }
                };
                let _ = std::fs::remove_file(&temp_path); // Clean up temp file

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
        // A text or categorical leaf counts its pages where they are more
        // (long strings, a dictionary of long labels); a fixed-width leaf
        // never does (its pages hold levels its values in memory do not).
        assert_eq!(least_with_pages(&DataType::String, 2, &[2], 1000), 1000);
        assert_eq!(
            least_with_pages(&list(DataType::Binary), 1, &[3], 1000),
            8 + 1000
        );
        assert_eq!(least_with_pages(&DataType::Float64, 2, &[2], 1000), 16);
        assert_eq!(least_with_pages(&DataType::String, 200, &[200], 1000), 3200);
    }

    /// What the quotas charge for `df` once written to parquet and loaded
    /// as `d`, and the least [`parquet_decoded_bytes`] gave for it before.
    fn loaded_charge_and_bound(df: &mut DataFrame) -> (u64, u64) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.parquet");
        p2a_core::polars::prelude::ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(df)
            .unwrap();
        let bound = parquet_decoded_bytes(&path).unwrap();
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
        let (charge, bound) = loaded_charge_and_bound(&mut df);
        assert!(bound <= charge, "bound {bound} > charge {charge}");
        assert!(
            bound >= 8 * inner_floats,
            "bound {bound}, {inner_floats} list values"
        );

        // Each column alone, too.
        for name in ["lists", "nested", "structs", "cats", "deep", "arr"] {
            let mut one = df.select([name]).unwrap();
            let (charge, bound) = loaded_charge_and_bound(&mut one);
            assert!(bound <= charge, "{name}: bound {bound} > charge {charge}");
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
        let honest = parquet_decoded_bytes(&flat).unwrap();
        assert!(honest >= 8 * 300_000, "{honest}");
        rewrite_footer(&flat, |footer| footer.num_rows = 1);
        assert_eq!(parquet_decoded_bytes(&flat), Some(honest));
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
        let honest = parquet_decoded_bytes(&lists).unwrap();
        assert!(honest >= 8 * 499_999, "{honest}");
        rewrite_footer(&lists, |footer| {
            for chunk in footer.row_groups.iter_mut().flat_map(|g| &mut g.columns) {
                let meta = chunk.meta_data.as_mut().unwrap();
                meta.num_values = 1;
                // A footer can understate the pages' size too.
                meta.total_uncompressed_size = 1;
            }
        });
        assert_eq!(parquet_decoded_bytes(&lists), Some(honest));
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
        let bound = parquet_decoded_bytes(&text).unwrap();
        assert!(bound >= 2_000 * 1_000, "{bound}");
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
        let file = std::fs::File::open(&path).unwrap();
        let file_len = file.metadata().unwrap().len();
        let mut pages = std::io::BufReader::new(file);
        let mut chunks = 0;
        for group in &metadata.row_groups {
            for chunk in group.parquet_columns() {
                let read =
                    read_chunk_pages(&mut pages, chunk.byte_range(), chunk.num_values(), file_len);
                assert_eq!(read.entries, chunk.num_values() as u64);
                assert!(read.page_bytes > 0);
                assert!(read.page_bytes <= chunk.uncompressed_size() as u64);
                chunks += 1;
            }
        }
        assert!(metadata.row_groups.len() > 1);
        assert_eq!(chunks, metadata.row_groups.len() * 3);
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
