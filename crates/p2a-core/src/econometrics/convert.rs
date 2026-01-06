//! Conversion utilities between Polars and greeners DataFrames.

use anyhow::{anyhow, Result};
use greeners::DataFrame as GreenersDataFrame;
use polars::prelude::*;

/// Convert a Polars DataFrame to a greeners DataFrame.
///
/// This function extracts all columns and rebuilds them in greeners format.
/// Numeric columns are converted to f64, string columns to String.
pub fn polars_to_greeners(df: &polars::frame::DataFrame) -> Result<GreenersDataFrame> {
    let mut builder = GreenersDataFrame::builder();

    for col in df.get_columns() {
        let name = col.name().to_string();
        let dtype = col.dtype();

        match dtype {
            DataType::Float64 => {
                let values: Vec<f64> = col
                    .f64()
                    .map_err(|e| anyhow!("Failed to read f64 column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.unwrap_or(f64::NAN))
                    .collect();
                builder = builder.add_column(&name, values);
            }
            DataType::Float32 => {
                let values: Vec<f64> = col
                    .f32()
                    .map_err(|e| anyhow!("Failed to read f32 column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
                    .collect();
                builder = builder.add_column(&name, values);
            }
            DataType::Int64 => {
                let values: Vec<f64> = col
                    .i64()
                    .map_err(|e| anyhow!("Failed to read i64 column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
                    .collect();
                builder = builder.add_column(&name, values);
            }
            DataType::Int32 => {
                let values: Vec<f64> = col
                    .i32()
                    .map_err(|e| anyhow!("Failed to read i32 column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
                    .collect();
                builder = builder.add_column(&name, values);
            }
            DataType::UInt64 => {
                let values: Vec<f64> = col
                    .u64()
                    .map_err(|e| anyhow!("Failed to read u64 column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
                    .collect();
                builder = builder.add_column(&name, values);
            }
            DataType::UInt32 => {
                let values: Vec<f64> = col
                    .u32()
                    .map_err(|e| anyhow!("Failed to read u32 column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
                    .collect();
                builder = builder.add_column(&name, values);
            }
            DataType::String => {
                let values: Vec<String> = col
                    .str()
                    .map_err(|e| anyhow!("Failed to read string column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.unwrap_or("").to_string())
                    .collect();
                builder = builder.add_string(&name, values);
            }
            DataType::Boolean => {
                // Convert boolean to 0/1
                let values: Vec<f64> = col
                    .bool()
                    .map_err(|e| anyhow!("Failed to read bool column '{}': {}", name, e))?
                    .into_iter()
                    .map(|v| v.map(|b| if b { 1.0 } else { 0.0 }).unwrap_or(f64::NAN))
                    .collect();
                builder = builder.add_column(&name, values);
            }
            _ => {
                // Try to cast to f64
                if let Ok(casted) = col.cast(&DataType::Float64) {
                    let values: Vec<f64> = casted
                        .f64()
                        .map_err(|e| anyhow!("Failed to cast column '{}' to f64: {}", name, e))?
                        .into_iter()
                        .map(|v| v.unwrap_or(f64::NAN))
                        .collect();
                    builder = builder.add_column(&name, values);
                } else {
                    // Skip unsupported column types
                    tracing::warn!("Skipping unsupported column type for '{}': {:?}", name, dtype);
                }
            }
        }
    }

    builder.build().map_err(|e| anyhow!("Failed to build greeners DataFrame: {}", e))
}
