//! Data loading functionality for various file formats.

#[cfg(feature = "file-formats")]
use calamine::{Data, Range, Reader, open_workbook_auto};
#[cfg(feature = "file-formats")]
use polars::frame::column::Column;
use polars::prelude::*;
use std::path::{Path, PathBuf};
use thiserror::Error;

use super::Dataset;

/// Errors that can occur during data loading.
#[derive(Error, Debug)]
pub enum LoadError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Failed to read file: {0}")]
    PolarsError(#[from] PolarsError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[cfg(feature = "file-formats")]
    #[error("Excel error: {0}")]
    ExcelError(String),

    #[cfg(not(feature = "file-formats"))]
    #[error("Excel support requires 'file-formats' feature")]
    ExcelNotSupported,

    #[error("Stata error: {0}")]
    StataError(#[from] super::stata::StataError),

    #[error("SAS error: {0}")]
    SasError(#[from] super::sas::SasError),

    /// A parquet column whose schema polars cannot turn into a column type.
    #[error("Cannot read column '{column}': {reason}")]
    ParquetSchema { column: String, reason: String },
}

/// Refuse a parquet file whose schema polars' conversion to column types
/// (`DataType::from_arrow_field`) would panic on, naming the column.
///
/// polars keeps a Categorical or Enum column's type in the field metadata
/// of the file's ARROW:schema, as text it parses with `unwrap` and
/// `split_at`, so malformed text in a file panics the reader. This checks
/// that text the way polars 0.52 parses it, in every field it would reach
/// (nested list, struct and map fields included), before the reader
/// converts it; a file polars wrote passes.
fn check_parquet_schema(schema: &ArrowSchema) -> Result<(), LoadError> {
    for field in schema.iter_values() {
        check_dtype(&field.dtype, field.metadata.as_deref()).map_err(|reason| {
            LoadError::ParquetSchema {
                column: field.name.to_string(),
                reason,
            }
        })?;
    }
    Ok(())
}

/// [`check_parquet_schema`] of a field of type `dtype` with the metadata
/// `md` and the fields nested in it, following `DataType::from_arrow` arm
/// by arm where it reads metadata or recurses.
fn check_dtype<M: MetaDataExt>(dtype: &ArrowDataType, md: Option<&M>) -> Result<(), String> {
    match dtype {
        // The enum metadata wins over the categorical one, as in polars;
        // a dictionary of other values is read as its values, without
        // metadata.
        ArrowDataType::Dictionary(_, value, _) => {
            if let Some(labels) = md.and_then(|md| md.pl_enum_metadata()) {
                check_enum_metadata(labels)
            } else if let Some(categories) = md.and_then(|md| md.pl_categorical_metadata()) {
                check_categorical_metadata(categories)
            } else {
                check_dtype::<M>(value, None)
            }
        }
        ArrowDataType::List(inner)
        | ArrowDataType::LargeList(inner)
        | ArrowDataType::FixedSizeList(inner, _)
        | ArrowDataType::Map(inner, _) => check_nested(inner),
        ArrowDataType::Struct(fields) => fields.iter().try_for_each(check_nested),
        _ => Ok(()),
    }
}

/// [`check_dtype`] of the field `inner` nested in another, whose reason
/// names it.
fn check_nested(inner: &ArrowField) -> Result<(), String> {
    check_dtype(&inner.dtype, inner.metadata.as_deref())
        .map_err(|reason| format!("its nested field '{}': {reason}", shown(&inner.name)))
}

/// At most 40 characters of `text` from a file, for an error message.
fn shown(text: &str) -> String {
    match text.char_indices().nth(40) {
        Some((at, _)) => format!("{}...", &text[..at]),
        None => text.to_string(),
    }
}

/// The text after a length `len` (in bytes) of `text`, or why there is
/// none: polars takes it with `split_at`, which panics when `len` runs past
/// the end or into a character.
fn after_length<'a>(text: &'a str, len: &str, what: &str) -> Result<(&'a str, &'a str), String> {
    let len: usize = len
        .parse()
        .map_err(|_| format!("the {what} length '{}' is not a number", shown(len)))?;
    match (text.get(..len), text.get(len..)) {
        (Some(head), Some(rest)) => Ok((head, rest)),
        _ => Err(format!(
            "the {what} length {len} runs past the end of the metadata or into a character"
        )),
    }
}

/// Check polars' Enum metadata `labels`: each label as its length in bytes,
/// `;` and the label, the labels distinct (polars builds its frozen
/// categories from them and unwraps the result).
fn check_enum_metadata(mut labels: &str) -> Result<(), String> {
    let malformed = |what: String| format!("its polars enum metadata is malformed: {what}");
    let mut seen = std::collections::HashSet::new();
    while !labels.is_empty() {
        let (len, rest) = labels
            .split_once(';')
            .ok_or_else(|| malformed(format!("no label length before '{}'", shown(labels))))?;
        let (label, rest) = after_length(rest, len, "label").map_err(malformed)?;
        if !seen.insert(label) {
            return Err(malformed(format!(
                "the label '{}' appears twice",
                shown(label)
            )));
        }
        labels = rest;
    }
    // polars refuses (and unwraps) as many labels as a u32 code can hold.
    if seen.len() >= u32::MAX as usize {
        return Err(malformed("more labels than a u32 code holds".to_string()));
    }
    Ok(())
}

/// Check polars' Categorical metadata `categories`: the categories' name
/// and namespace, each as its length in bytes, `;` and the text, then the
/// code width (`u8`, `u16` or `u32`) and `;`.
fn check_categorical_metadata(categories: &str) -> Result<(), String> {
    let malformed = |what: String| format!("its polars categorical metadata is malformed: {what}");
    let mut rest = categories;
    for what in ["name", "namespace"] {
        let (len, after) = rest
            .split_once(';')
            .ok_or_else(|| malformed(format!("no {what} length")))?;
        (_, rest) = after_length(after, len, what).map_err(malformed)?;
    }
    let (physical, _) = rest
        .split_once(';')
        .ok_or_else(|| malformed("no code width".to_string()))?;
    physical.parse::<CategoricalPhysical>().map_err(|()| {
        malformed(format!(
            "the code width '{}' is not u8, u16 or u32",
            shown(physical)
        ))
    })?;
    Ok(())
}

/// Read a whole parquet file from `reader`, refusing one whose schema
/// [`check_parquet_schema`] refuses before it is decoded.
fn read_parquet<R: polars::io::mmap::MmapBytesReader>(reader: R) -> Result<DataFrame, LoadError> {
    let mut reader = ParquetReader::new(reader);
    // The reader keeps the schema it reads here for `finish`.
    let schema = reader.schema()?;
    check_parquet_schema(&schema)?;
    Ok(reader.finish()?)
}

/// Data loader supporting multiple file formats.
pub struct DataLoader;

impl DataLoader {
    /// Load a dataset from a file path.
    ///
    /// The file format is automatically detected from the extension.
    pub fn load(path: impl AsRef<Path>) -> Result<Dataset, LoadError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(LoadError::FileNotFound(path.display().to_string()));
        }

        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        let df = match extension.as_str() {
            "csv" => Self::load_csv(path)?,
            "parquet" | "pq" => Self::load_parquet(path)?,
            #[cfg(feature = "file-formats")]
            "xlsx" | "xls" | "xlsb" | "ods" => Self::load_excel(path, None)?,
            #[cfg(not(feature = "file-formats"))]
            "xlsx" | "xls" | "xlsb" | "ods" => return Err(LoadError::ExcelNotSupported),
            "dta" => Self::load_stata(path)?,
            "sas7bdat" => Self::load_sas(path)?,
            _ => {
                #[cfg(feature = "file-formats")]
                let supported = "csv, parquet, xlsx, xls, xlsb, ods, dta, sas7bdat";
                #[cfg(not(feature = "file-formats"))]
                let supported = "csv, parquet, dta, sas7bdat (enable 'file-formats' feature for xlsx/xls/xlsb/ods)";
                return Err(LoadError::UnsupportedFormat(format!(
                    "Extension '{}' not supported. Supported formats: {}",
                    extension, supported
                )));
            }
        };

        let name = path.file_stem().and_then(|s| s.to_str()).map(String::from);

        let mut dataset = Dataset::new(df).with_source_path(path);
        if let Some(name) = name {
            dataset = dataset.with_name(name);
        }

        Ok(dataset)
    }

    /// Load a CSV file eagerly (entire file into memory).
    pub fn load_csv(path: impl AsRef<Path>) -> Result<DataFrame, LoadError> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(1000))
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        Ok(df)
    }

    /// Load a Parquet file eagerly (entire file into memory).
    ///
    /// A file whose polars type metadata is malformed is refused with
    /// [`LoadError::ParquetSchema`] naming the column, before it is decoded.
    pub fn load_parquet(path: impl AsRef<Path>) -> Result<DataFrame, LoadError> {
        let file = std::fs::File::open(path.as_ref())?;
        read_parquet(file)
    }

    /// Load a Parquet file held in memory (an upload, say) eagerly, as
    /// [`DataLoader::load_parquet`] loads one from a path.
    pub fn load_parquet_bytes(bytes: &[u8]) -> Result<DataFrame, LoadError> {
        read_parquet(std::io::Cursor::new(bytes))
    }

    /// Load a CSV file with optional row limit (useful for previewing large files).
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `n_rows` - Optional maximum number of rows to read
    ///
    /// # Example
    /// ```ignore
    /// // Preview first 1000 rows of a large file
    /// let df = DataLoader::load_csv_with_limit("large_file.csv", Some(1000))?;
    /// ```
    pub fn load_csv_with_limit(
        path: impl AsRef<Path>,
        n_rows: Option<usize>,
    ) -> Result<DataFrame, LoadError> {
        let mut options = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(1000));

        if let Some(limit) = n_rows {
            options = options.with_n_rows(Some(limit));
        }

        let df = options
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        Ok(df)
    }

    /// Load a Parquet file with optional row limit.
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    /// * `n_rows` - Optional maximum number of rows to read
    ///
    /// # Example
    /// ```ignore
    /// // Preview first 1000 rows of a large Parquet file
    /// let df = DataLoader::load_parquet_with_limit("large_file.parquet", Some(1000))?;
    /// ```
    pub fn load_parquet_with_limit(
        path: impl AsRef<Path>,
        n_rows: Option<usize>,
    ) -> Result<DataFrame, LoadError> {
        let file = std::fs::File::open(path.as_ref())?;
        let df = if let Some(limit) = n_rows {
            // Read entire file but take only first n rows
            // Note: For very large files, consider using streaming readers
            let full_df = read_parquet(file)?;
            full_df.head(Some(limit))
        } else {
            read_parquet(file)?
        };
        Ok(df)
    }

    /// Load CSV in chunks for processing large files.
    ///
    /// Reads a specific chunk (slice) of a CSV file.
    /// Use this to process large files in pieces without loading everything.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `skip_rows` - Number of rows to skip (excluding header)
    /// * `n_rows` - Number of rows to read
    ///
    /// # Example
    /// ```ignore
    /// // Process a large CSV file in 10,000 row chunks
    /// let chunk_size = 10_000;
    /// let mut offset = 0;
    /// loop {
    ///     let chunk = DataLoader::load_csv_chunk("large_file.csv", offset, chunk_size)?;
    ///     if chunk.height() == 0 { break; }
    ///     // Process chunk...
    ///     offset += chunk_size;
    /// }
    /// ```
    pub fn load_csv_chunk(
        path: impl AsRef<Path>,
        skip_rows: usize,
        n_rows: usize,
    ) -> Result<DataFrame, LoadError> {
        let options = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(1000))
            .with_skip_rows(skip_rows)
            .with_n_rows(Some(n_rows));

        let df = options
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        Ok(df)
    }

    /// Create a chunk iterator for processing large CSV files.
    ///
    /// Returns an iterator that yields DataFrames in chunks of the specified size.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `chunk_size` - Number of rows per chunk
    pub fn iter_csv_chunks(path: impl AsRef<Path>, chunk_size: usize) -> CsvChunkIterator {
        CsvChunkIterator {
            path: path.as_ref().to_path_buf(),
            chunk_size,
            current_offset: 0,
            exhausted: false,
        }
    }

    /// Get file metadata without loading the entire file.
    ///
    /// Useful for estimating file size before loading.
    pub fn file_info(path: impl AsRef<Path>) -> Result<FileInfo, LoadError> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)?;
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        // For CSV, read just the header to get column count
        let (n_columns, column_names) = if extension == "csv" {
            let df = Self::load_csv_with_limit(path, Some(1))?;
            let names: Vec<String> = df
                .get_column_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            (names.len(), Some(names))
        } else {
            (0, None)
        };

        Ok(FileInfo {
            path: path.to_path_buf(),
            size_bytes: metadata.len() as usize,
            format: extension,
            n_columns,
            column_names,
        })
    }
}

/// Iterator for reading CSV files in chunks.
pub struct CsvChunkIterator {
    path: PathBuf,
    chunk_size: usize,
    current_offset: usize,
    exhausted: bool,
}

impl Iterator for CsvChunkIterator {
    type Item = Result<DataFrame, LoadError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        match DataLoader::load_csv_chunk(&self.path, self.current_offset, self.chunk_size) {
            Ok(df) => {
                if df.height() == 0 {
                    self.exhausted = true;
                    None
                } else {
                    self.current_offset += df.height();
                    // Mark exhausted if we got fewer rows than requested
                    if df.height() < self.chunk_size {
                        self.exhausted = true;
                    }
                    Some(Ok(df))
                }
            }
            Err(e) => {
                self.exhausted = true;
                Some(Err(e))
            }
        }
    }
}

/// Information about a data file without loading its contents.
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// Path to the file
    pub path: PathBuf,
    /// File size in bytes
    pub size_bytes: usize,
    /// File format (extension)
    pub format: String,
    /// Number of columns (if determinable)
    pub n_columns: usize,
    /// Column names (if determinable)
    pub column_names: Option<Vec<String>>,
}

impl FileInfo {
    /// Format file size as human-readable string.
    pub fn size_formatted(&self) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;

        if self.size_bytes >= GB {
            format!("{:.2} GB", self.size_bytes as f64 / GB as f64)
        } else if self.size_bytes >= MB {
            format!("{:.2} MB", self.size_bytes as f64 / MB as f64)
        } else if self.size_bytes >= KB {
            format!("{:.2} KB", self.size_bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", self.size_bytes)
        }
    }

    /// Estimate if file is "large" (> 100 MB).
    pub fn is_large(&self) -> bool {
        self.size_bytes > 100 * 1024 * 1024
    }
}

impl DataLoader {
    /// Load an Excel file (xlsx, xls, xlsb, ods).
    ///
    /// # Arguments
    /// * `path` - Path to the Excel file
    /// * `sheet_name` - Optional sheet name (uses first sheet if None)
    ///
    /// Requires the `file-formats` feature.
    #[cfg(feature = "file-formats")]
    pub fn load_excel(
        path: impl AsRef<Path>,
        sheet_name: Option<&str>,
    ) -> Result<DataFrame, LoadError> {
        let path = path.as_ref();

        let mut workbook = open_workbook_auto(path)
            .map_err(|e| LoadError::ExcelError(format!("Failed to open workbook: {}", e)))?;

        // Get sheet name - use provided name or first sheet
        let sheet_to_use = match sheet_name {
            Some(name) => name.to_string(),
            None => {
                let sheets = workbook.sheet_names();
                if sheets.is_empty() {
                    return Err(LoadError::ExcelError("Workbook has no sheets".to_string()));
                }
                sheets[0].clone()
            }
        };

        // Read the worksheet
        let range: Range<Data> = workbook.worksheet_range(&sheet_to_use).map_err(|e| {
            LoadError::ExcelError(format!("Failed to read sheet '{}': {}", sheet_to_use, e))
        })?;

        // Convert to DataFrame
        excel_range_to_dataframe(&range)
    }

    /// Load a Stata DTA file.
    pub fn load_stata(path: impl AsRef<Path>) -> Result<DataFrame, LoadError> {
        let df = super::stata::load_stata(path)?;
        Ok(df)
    }

    /// Load a SAS7BDAT file.
    pub fn load_sas(path: impl AsRef<Path>) -> Result<DataFrame, LoadError> {
        let df = super::sas::load_sas(path)?;
        Ok(df)
    }

    /// Load data from a CSV string.
    pub fn from_csv_string(data: &str) -> Result<DataFrame, LoadError> {
        let cursor = std::io::Cursor::new(data);
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(1000))
            .into_reader_with_file_handle(cursor)
            .finish()?;
        Ok(df)
    }
}

/// Convert a calamine Range to a Polars DataFrame.
///
/// Assumes the first row contains column headers.
#[cfg(feature = "file-formats")]
fn excel_range_to_dataframe(range: &Range<Data>) -> Result<DataFrame, LoadError> {
    let (height, width) = range.get_size();

    if height == 0 || width == 0 {
        return Err(LoadError::ExcelError("Empty worksheet".to_string()));
    }

    // Extract headers from first row
    let mut headers: Vec<String> = Vec::with_capacity(width);
    for col_idx in 0..width {
        let header = match range.get((0, col_idx)) {
            Some(data) => data_to_string(data),
            None => format!("col_{}", col_idx),
        };
        headers.push(header);
    }

    // Extract data columns - need to determine types
    // First pass: collect all values as strings, then infer types
    let mut columns: Vec<Vec<Option<String>>> = vec![Vec::with_capacity(height - 1); width];

    for row_idx in 1..height {
        for col_idx in 0..width {
            let value = match range.get((row_idx, col_idx)) {
                Some(data) => data_to_option_string(data),
                None => None,
            };
            columns[col_idx].push(value);
        }
    }

    // Build columns, attempting to infer numeric types
    let mut column_vec: Vec<Column> = Vec::with_capacity(width);

    for (col_idx, col_data) in columns.iter().enumerate() {
        let name = &headers[col_idx];

        // Try to parse as numeric first
        let (is_all_numeric, is_all_int) = check_numeric_type(col_data);

        let column = if is_all_int {
            // Parse as integers
            let values: Vec<Option<i64>> = col_data
                .iter()
                .map(|opt| opt.as_ref().and_then(|s| s.parse::<i64>().ok()))
                .collect();
            Column::new(name.as_str().into(), values)
        } else if is_all_numeric {
            // Parse as floats
            let values: Vec<Option<f64>> = col_data
                .iter()
                .map(|opt| opt.as_ref().and_then(|s| s.parse::<f64>().ok()))
                .collect();
            Column::new(name.as_str().into(), values)
        } else {
            // Keep as strings
            let values: Vec<Option<&str>> = col_data.iter().map(|opt| opt.as_deref()).collect();
            Column::new(name.as_str().into(), values)
        };

        column_vec.push(column);
    }

    DataFrame::new(column_vec).map_err(LoadError::PolarsError)
}

/// Convert calamine Data to a string representation.
#[cfg(feature = "file-formats")]
fn data_to_string(data: &Data) -> String {
    match data {
        Data::Int(i) => i.to_string(),
        Data::Float(f) => f.to_string(),
        Data::String(s) => s.clone(),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => format!("{}", dt),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("#{:?}", e),
        Data::Empty => String::new(),
    }
}

/// Convert calamine Data to an optional string.
#[cfg(feature = "file-formats")]
fn data_to_option_string(data: &Data) -> Option<String> {
    match data {
        Data::Empty => None,
        Data::Error(_) => None,
        other => Some(data_to_string(other)),
    }
}

/// Check if all non-null values in a column are numeric, and if they're all integers.
#[cfg(feature = "file-formats")]
fn check_numeric_type(col_data: &[Option<String>]) -> (bool, bool) {
    let mut is_all_numeric = true;
    let mut is_all_int = true;
    let mut has_non_null = false;

    for opt in col_data {
        if let Some(s) = opt {
            has_non_null = true;

            // Check if it parses as a number
            if s.parse::<f64>().is_ok() {
                // Check if it's an integer (no decimal point or ends with .0)
                if s.parse::<i64>().is_err() && !s.ends_with(".0") {
                    is_all_int = false;
                }
            } else {
                is_all_numeric = false;
                is_all_int = false;
                break;
            }
        }
    }

    // If we have no non-null values, default to string
    if !has_non_null {
        return (false, false);
    }

    (is_all_numeric, is_all_int && is_all_numeric)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_csv_string() {
        let csv_data = "a,b,c\n1,2,3\n4,5,6";
        let df = DataLoader::from_csv_string(csv_data).unwrap();
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 3);
    }

    #[test]
    #[cfg(feature = "file-formats")]
    fn test_load_excel() {
        // Skip if test file doesn't exist
        let path = std::path::Path::new("tests/data/test.xlsx");
        if !path.exists() {
            return;
        }

        let dataset = DataLoader::load(path).unwrap();
        let df = dataset.df();

        // Test file has 3 rows of data (4 rows total including header)
        assert_eq!(df.height(), 3);
        // 3 columns: id, name, value
        assert_eq!(df.width(), 3);
    }

    /// Write `df` as a parquet file `name` in `dir`.
    fn parquet_file(dir: &Path, name: &str, mut df: DataFrame) -> PathBuf {
        let path = dir.join(name);
        ParquetWriter::new(std::fs::File::create(&path).unwrap())
            .finish(&mut df)
            .unwrap();
        path
    }

    /// A frame of the column `k` beside a noisy float column `x`.
    fn frame_of(k: Series) -> DataFrame {
        let x: Vec<f64> = (0..k.len()).map(|i| i as f64 * 1.1 + 0.3).collect();
        DataFrame::new(vec![Series::new("x".into(), x).into(), k.into()]).unwrap()
    }

    /// A frame of a column `k` of `labels` (a row each) cast to `dtype`.
    fn frame_with(dtype: &DataType, labels: &[&str]) -> DataFrame {
        frame_of(Series::new("k".into(), labels).cast(dtype).unwrap())
    }

    /// A frame of a column `k` of one-row lists of `labels` cast to lists of
    /// `dtype`.
    fn list_frame_with(dtype: &DataType, labels: &[&str]) -> DataFrame {
        let lists: Vec<Series> = labels
            .iter()
            .map(|label| Series::new("".into(), [*label]))
            .collect();
        let k = Series::new("k".into(), lists)
            .cast(&DataType::List(Box::new(dtype.clone())))
            .unwrap();
        frame_of(k)
    }

    /// A frame of a column `k` of structs of one field `c` of `labels` cast
    /// to `dtype`.
    fn struct_frame_with(dtype: &DataType, labels: &[&str]) -> DataFrame {
        let c = Series::new("c".into(), labels).cast(dtype).unwrap();
        let k = StructChunked::from_series("k".into(), c.len(), [c].iter()).unwrap();
        frame_of(k.into_series())
    }

    /// A Categorical column type of polars' categories `name` in the default
    /// namespace with `physical` codes.
    fn categorical(name: &str, physical: CategoricalPhysical) -> DataType {
        DataType::from_categories(Categories::new(name.into(), "".into(), physical))
    }

    /// An Enum column type of `labels`.
    fn enumeration(labels: &[&str]) -> DataType {
        DataType::from_frozen_categories(FrozenCategories::new(labels.iter().copied()).unwrap())
    }

    /// Rewrite the base64 ARROW:schema of the parquet file at `path` in
    /// place, with `from` replaced by `to` (of the same length) in its
    /// decoded IPC bytes: the base64 text keeps its length, so the footer
    /// keeps its layout and still parses.
    fn edit_arrow_schema(path: &Path, from: &[u8], to: &[u8]) {
        use base64::Engine;
        assert_eq!(from.len(), to.len());
        let engine = base64::engine::general_purpose::STANDARD;
        let encoded = {
            let mut reader = ParquetReader::new(std::fs::File::open(path).unwrap());
            let metadata = reader.get_metadata().unwrap();
            let kv = metadata.key_value_metadata().as_ref().unwrap();
            let schema = kv.iter().find(|kv| kv.key == "ARROW:schema").unwrap();
            schema.value.clone().unwrap()
        };
        let mut ipc = engine.decode(&encoded).unwrap();
        let at = ipc.windows(from.len()).position(|w| w == from).unwrap();
        ipc[at..at + to.len()].copy_from_slice(to);
        let edited = engine.encode(&ipc);
        assert_eq!(edited.len(), encoded.len());
        let mut bytes = std::fs::read(path).unwrap();
        let at = bytes
            .windows(encoded.len())
            .position(|w| w == encoded.as_bytes())
            .unwrap();
        bytes[at..at + edited.len()].copy_from_slice(edited.as_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    /// The text of the error loading the parquet file at `path` every way
    /// the loader reads one, which must agree.
    fn parquet_error(path: &Path) -> String {
        let Err(direct) = DataLoader::load_parquet(path) else {
            panic!("{path:?} loaded");
        };
        let Err(by_extension) = DataLoader::load(path) else {
            panic!("{path:?} loaded by its extension");
        };
        let Err(limited) = DataLoader::load_parquet_with_limit(path, Some(1)) else {
            panic!("{path:?} loaded with a row limit");
        };
        let Err(in_memory) = DataLoader::load_parquet_bytes(&std::fs::read(path).unwrap()) else {
            panic!("{path:?} loaded from its bytes");
        };
        let text = direct.to_string();
        assert_eq!(by_extension.to_string(), text);
        assert_eq!(limited.to_string(), text);
        assert_eq!(in_memory.to_string(), text);
        text
    }

    /// The error loading `df` written as a parquet file whose ARROW:schema
    /// has `from` replaced by `to`. It names the column `k`.
    fn refusal_of(df: DataFrame, from: &str, to: &str) -> String {
        let dir = tempfile::tempdir().unwrap();
        let path = parquet_file(dir.path(), "edited.parquet", df);
        edit_arrow_schema(&path, from.as_bytes(), to.as_bytes());
        let text = parquet_error(&path);
        assert!(text.contains("column 'k'"), "{text}");
        text
    }

    #[test]
    fn parquet_categorical_metadata_polars_panics_on_is_refused_naming_the_column() {
        // polars keeps a Categorical column's categories (name, namespace
        // and code width) in its ARROW:schema metadata, here "3;lab0;u32;",
        // and parses it with unwrap and split_at. Pre-fix each of these
        // edits panicked the reader in polars' schema conversion.
        let lab = || frame_with(&categorical("lab", CategoricalPhysical::U32), &["lo", "hi"]);
        for (to, says) in [
            ("3;lab0;u99;", "the code width 'u99' is not u8, u16 or u32"),
            ("x;lab0;u32;", "the name length 'x' is not a number"),
            ("99;ab0;u32;", "the name length 99 runs past the end"),
            ("3;lab9;u32;", "the namespace length 9 runs past the end"),
            ("3;lab0;u32:", "no code width"),
            ("3:lab0:u32:", "no name length"),
        ] {
            let text = refusal_of(lab(), "3;lab0;u32;", to);
            assert!(
                text.contains("polars categorical metadata is malformed"),
                "{to}: {text}"
            );
            assert!(text.contains(says), "{to}: {text}");
        }
        // A length that ends inside a character ('é' is two bytes).
        let accented = frame_with(&categorical("éa", CategoricalPhysical::U32), &["lo"]);
        let text = refusal_of(accented, "3;éa0;", "1;éa0;");
        assert!(
            text.contains(
                "the name length 1 runs past the end of the metadata or into a character"
            ),
            "{text}"
        );
    }

    #[test]
    fn parquet_enum_metadata_polars_panics_on_is_refused_naming_the_column() {
        // polars keeps an Enum column's labels in its ARROW:schema metadata,
        // here "2;lo2;hi", and parses them with unwrap and split_at into
        // categories that must be distinct. Pre-fix each of these edits
        // panicked the reader in polars' schema conversion.
        let lohi = || frame_with(&enumeration(&["lo", "hi"]), &["lo", "hi"]);
        for (to, says) in [
            ("2;lo2;lo", "the label 'lo' appears twice"),
            ("2;lo9;hi", "the label length 9 runs past the end"),
            ("2;lox;hi", "the label length 'x' is not a number"),
            ("2;lo2:hi", "no label length before '2:hi'"),
        ] {
            let text = refusal_of(lohi(), "2;lo2;hi", to);
            assert!(
                text.contains("polars enum metadata is malformed"),
                "{to}: {text}"
            );
            assert!(text.contains(says), "{to}: {text}");
        }
        let accented = frame_with(&enumeration(&["é", "hi"]), &["é", "hi"]);
        let text = refusal_of(accented, "2;é2;hi", "1;é2;hi");
        assert!(
            text.contains(
                "the label length 1 runs past the end of the metadata or into a character"
            ),
            "{text}"
        );
    }

    #[test]
    fn malformed_metadata_of_a_nested_field_is_refused_naming_the_column_and_field() {
        // polars converts the type of each field nested in a list or a
        // struct the same way, so the same metadata panicked it there.
        let dtype = categorical("lab", CategoricalPhysical::U32);
        let text = refusal_of(list_frame_with(&dtype, &["lo", "hi"]), "u32;", "u99;");
        assert!(text.contains("its nested field '"), "{text}");
        assert!(text.contains("the code width 'u99'"), "{text}");
        let text = refusal_of(struct_frame_with(&dtype, &["lo", "hi"]), "u32;", "u99;");
        assert!(text.contains("its nested field 'c'"), "{text}");
        assert!(text.contains("the code width 'u99'"), "{text}");
    }

    #[test]
    fn categorical_and_enum_columns_polars_writes_still_load() {
        // The check passes the metadata polars writes, at the top level and
        // nested, and the columns load as written.
        let dir = tempfile::tempdir().unwrap();
        let labels = ["lo", "hi", "lo", "mid"];
        let cat = categorical("lab", CategoricalPhysical::U32);
        let narrow = categorical("nar", CategoricalPhysical::U8);
        let accented = categorical("éa", CategoricalPhysical::U16);
        let lohi = enumeration(&["lo", "mid", "hi"]);
        for (name, df) in [
            ("categorical", frame_with(&cat, &labels)),
            ("narrow", frame_with(&narrow, &labels)),
            ("accented", frame_with(&accented, &labels)),
            ("enum", frame_with(&lohi, &labels)),
            ("list", list_frame_with(&cat, &labels)),
            ("struct", struct_frame_with(&lohi, &labels)),
        ] {
            let path = parquet_file(dir.path(), &format!("{name}.parquet"), df.clone());
            let loaded = DataLoader::load_parquet(&path).unwrap();
            assert_eq!(loaded.schema(), df.schema(), "{name}");
            assert!(loaded.equals_missing(&df), "{name}: {loaded} vs {df}");
            let in_memory = DataLoader::load_parquet_bytes(&std::fs::read(&path).unwrap());
            assert!(in_memory.unwrap().equals_missing(&df), "{name}");
        }
    }
}
