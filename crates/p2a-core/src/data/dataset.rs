//! Dataset wrapper providing a convenient interface to Polars DataFrames.

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
        self.df.get_column_names().into_iter().map(|s| s.to_string()).collect()
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
