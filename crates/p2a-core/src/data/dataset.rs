//! Dataset wrapper providing a convenient interface to Polars DataFrames.

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};

/// A dataset wrapper around a Polars DataFrame with metadata.
#[derive(Clone)]
pub struct Dataset {
    /// The underlying Polars DataFrame
    df: DataFrame,
    /// Optional name/identifier for the dataset
    name: Option<String>,
    /// Original file path if loaded from file
    source_path: Option<PathBuf>,
}

impl Dataset {
    /// Create a new Dataset from a Polars DataFrame.
    pub fn new(df: DataFrame) -> Self {
        Self {
            df,
            name: None,
            source_path: None,
        }
    }

    /// Create a new Dataset with a name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the source path for the dataset.
    pub fn with_source_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.source_path = Some(path.into());
        self
    }

    /// Get a reference to the underlying DataFrame.
    pub fn df(&self) -> &DataFrame {
        &self.df
    }

    /// Get a mutable reference to the underlying DataFrame.
    pub fn df_mut(&mut self) -> &mut DataFrame {
        &mut self.df
    }

    /// Consume and return the underlying DataFrame.
    pub fn into_df(self) -> DataFrame {
        self.df
    }

    /// Get the dataset name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the source path.
    pub fn source_path(&self) -> Option<&PathBuf> {
        self.source_path.as_ref()
    }

    /// Get the number of rows.
    pub fn nrows(&self) -> usize {
        self.df.height()
    }

    /// Get the number of columns.
    pub fn ncols(&self) -> usize {
        self.df.width()
    }

    /// Get column names as strings.
    pub fn column_names(&self) -> Vec<String> {
        self.df
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the first n rows.
    pub fn head(&self, n: Option<usize>) -> DataFrame {
        self.df.head(n)
    }

    /// Get the last n rows.
    pub fn tail(&self, n: Option<usize>) -> DataFrame {
        self.df.tail(n)
    }

    /// Get a specific column by name.
    pub fn column(&self, name: &str) -> PolarsResult<&Column> {
        self.df.column(name)
    }

    /// Convert to a lazy frame for deferred computation.
    ///
    /// Lazy frames allow building up a query plan that is only executed
    /// when `collect()` is called, enabling query optimization.
    pub fn lazy(&self) -> LazyFrame {
        self.df.clone().lazy()
    }

    /// Sample n random rows from the dataset.
    ///
    /// Useful for quick analysis of large datasets.
    pub fn sample(&self, n: usize, seed: Option<u64>) -> PolarsResult<Dataset> {
        let df = self.df.sample_n_literal(n, false, false, seed)?;
        Ok(Dataset {
            df,
            name: self.name.clone(),
            source_path: self.source_path.clone(),
        })
    }

    /// Get a filtered subset of the dataset.
    ///
    /// Uses lazy evaluation for efficiency.
    pub fn filter(&self, predicate: Expr) -> PolarsResult<Dataset> {
        let df = self.df.clone().lazy().filter(predicate).collect()?;
        Ok(Dataset {
            df,
            name: self.name.clone(),
            source_path: self.source_path.clone(),
        })
    }

    /// Select specific columns from the dataset.
    ///
    /// Uses lazy evaluation for efficiency.
    pub fn select_columns(&self, columns: &[&str]) -> PolarsResult<Dataset> {
        let exprs: Vec<Expr> = columns.iter().map(|c| col(*c)).collect();
        let df = self.df.clone().lazy().select(exprs).collect()?;
        Ok(Dataset {
            df,
            name: self.name.clone(),
            source_path: self.source_path.clone(),
        })
    }

    /// Export the dataset to a CSV file.
    ///
    /// # Arguments
    /// * `path` - Path to the output CSV file
    ///
    /// # Example
    /// ```ignore
    /// dataset.to_csv("output.csv")?;
    /// ```
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> PolarsResult<()> {
        let file = File::create(path)?;
        let mut writer = CsvWriter::new(file);
        writer.finish(&mut self.df.clone())
    }

    /// Export the dataset to a CSV string.
    ///
    /// Useful for returning CSV data in API responses.
    pub fn to_csv_string(&self) -> PolarsResult<String> {
        let mut buf = Vec::new();
        {
            let mut writer = CsvWriter::new(&mut buf);
            writer.finish(&mut self.df.clone())?;
        }
        String::from_utf8(buf).map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }

    /// Export the dataset to a Parquet file.
    ///
    /// # Arguments
    /// * `path` - Path to the output Parquet file
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> PolarsResult<()> {
        let file = File::create(path)?;
        ParquetWriter::new(file).finish(&mut self.df.clone())?;
        Ok(())
    }

    /// Export the dataset to a JSON string (records format).
    ///
    /// Each row becomes a JSON object with column names as keys.
    pub fn to_json_string(&self) -> PolarsResult<String> {
        // Convert DataFrame to JSON manually since polars JSON features may not be enabled
        let mut records = Vec::new();
        let height = self.df.height();
        let cols = self.df.get_columns();

        for i in 0..height {
            let mut row = serde_json::Map::new();
            for col in cols {
                let val = col.get(i).map_or(serde_json::Value::Null, |v| {
                    // Convert AnyValue to serde_json::Value
                    match v {
                        polars::datatypes::AnyValue::Null => serde_json::Value::Null,
                        polars::datatypes::AnyValue::Boolean(b) => serde_json::Value::Bool(b),
                        polars::datatypes::AnyValue::Int8(n) => serde_json::Value::Number(n.into()),
                        polars::datatypes::AnyValue::Int16(n) => {
                            serde_json::Value::Number(n.into())
                        }
                        polars::datatypes::AnyValue::Int32(n) => {
                            serde_json::Value::Number(n.into())
                        }
                        polars::datatypes::AnyValue::Int64(n) => {
                            serde_json::Value::Number(n.into())
                        }
                        polars::datatypes::AnyValue::UInt8(n) => {
                            serde_json::Value::Number(n.into())
                        }
                        polars::datatypes::AnyValue::UInt16(n) => {
                            serde_json::Value::Number(n.into())
                        }
                        polars::datatypes::AnyValue::UInt32(n) => {
                            serde_json::Value::Number(n.into())
                        }
                        polars::datatypes::AnyValue::UInt64(n) => {
                            serde_json::Value::Number(n.into())
                        }
                        polars::datatypes::AnyValue::Float32(n) => {
                            serde_json::Number::from_f64(n as f64)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null)
                        }
                        polars::datatypes::AnyValue::Float64(n) => serde_json::Number::from_f64(n)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null),
                        _ => serde_json::Value::String(format!("{}", v)),
                    }
                });
                row.insert(col.name().to_string(), val);
            }
            records.push(serde_json::Value::Object(row));
        }

        serde_json::to_string_pretty(&records)
            .map_err(|e| PolarsError::ComputeError(e.to_string().into()))
    }
}

/// Serializable metadata about a dataset (for MCP responses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: Option<String>,
    pub source_path: Option<String>,
    pub nrows: usize,
    pub ncols: usize,
    pub columns: Vec<ColumnInfo>,
}

/// Information about a single column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    pub name: String,
    pub dtype: String,
    pub null_count: usize,
}

impl From<&Dataset> for DatasetInfo {
    fn from(dataset: &Dataset) -> Self {
        let columns = dataset
            .df()
            .get_columns()
            .iter()
            .map(|col| ColumnInfo {
                name: col.name().to_string(),
                dtype: col.dtype().to_string(),
                null_count: col.null_count(),
            })
            .collect();

        Self {
            name: dataset.name().map(String::from),
            source_path: dataset.source_path().map(|p| p.display().to_string()),
            nrows: dataset.nrows(),
            ncols: dataset.ncols(),
            columns,
        }
    }
}
