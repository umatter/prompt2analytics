//! Data loading functionality for various file formats.

#[cfg(feature = "file-formats")]
use calamine::{open_workbook_auto, Data, Range, Reader};
use polars::prelude::*;
#[cfg(feature = "file-formats")]
use polars::frame::column::Column;
use std::path::Path;
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
                )))
            }
        };

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(String::from);

        let mut dataset = Dataset::new(df).with_source_path(path);
        if let Some(name) = name {
            dataset = dataset.with_name(name);
        }

        Ok(dataset)
    }

    /// Load a CSV file.
    pub fn load_csv(path: impl AsRef<Path>) -> Result<DataFrame, LoadError> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(1000))
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;
        Ok(df)
    }

    /// Load a Parquet file.
    pub fn load_parquet(path: impl AsRef<Path>) -> Result<DataFrame, LoadError> {
        let file = std::fs::File::open(path.as_ref())?;
        let df = ParquetReader::new(file).finish()?;
        Ok(df)
    }

    /// Load an Excel file (xlsx, xls, xlsb, ods).
    ///
    /// # Arguments
    /// * `path` - Path to the Excel file
    /// * `sheet_name` - Optional sheet name (uses first sheet if None)
    ///
    /// Requires the `file-formats` feature.
    #[cfg(feature = "file-formats")]
    pub fn load_excel(path: impl AsRef<Path>, sheet_name: Option<&str>) -> Result<DataFrame, LoadError> {
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
        let range: Range<Data> = workbook
            .worksheet_range(&sheet_to_use)
            .map_err(|e| LoadError::ExcelError(format!("Failed to read sheet '{}': {}", sheet_to_use, e)))?;

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
            let values: Vec<Option<&str>> = col_data
                .iter()
                .map(|opt| opt.as_deref())
                .collect();
            Column::new(name.as_str().into(), values)
        };

        column_vec.push(column);
    }

    DataFrame::new(column_vec).map_err(|e| LoadError::PolarsError(e))
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
}
