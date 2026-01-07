//! Dataset management Tauri commands.

use crate::AppState;
use serde::{Deserialize, Serialize};
use tauri::State;

/// Dataset information returned to frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub column_names: Vec<String>,
    pub source_path: Option<String>,
}

/// Dataset preview with paginated rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetPreview {
    pub name: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub total_rows: usize,
    pub offset: usize,
    pub limit: usize,
}

/// List all loaded datasets.
#[tauri::command]
pub async fn list_datasets(state: State<'_, AppState>) -> Result<Vec<DatasetInfo>, String> {
    let client = state.mcp_client();

    if !client.is_running() {
        client.spawn().await.map_err(|e| e.to_string())?;
    }

    // Call list_datasets tool
    let result = client
        .call_tool("list_datasets", serde_json::json!({}))
        .await
        .map_err(|e| e.to_string())?;

    // Parse the text output to extract dataset info
    // The tool returns formatted text, so we need to parse it
    parse_dataset_list(&result.content)
}

/// Load a dataset from file.
#[tauri::command]
pub async fn load_dataset(
    state: State<'_, AppState>,
    path: String,
    name: Option<String>,
) -> Result<DatasetInfo, String> {
    let client = state.mcp_client();

    if !client.is_running() {
        client.spawn().await.map_err(|e| e.to_string())?;
    }

    let args = serde_json::json!({
        "path": path,
        "name": name
    });

    let result = client
        .call_tool("load_dataset", args)
        .await
        .map_err(|e| e.to_string())?;

    if !result.success {
        return Err(result.error.unwrap_or_else(|| "Unknown error".to_string()));
    }

    // Parse the load result
    parse_load_result(&result.content, &path)
}

/// Get a preview of dataset rows.
#[tauri::command]
pub async fn get_dataset_preview(
    state: State<'_, AppState>,
    dataset_name: String,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Result<DatasetPreview, String> {
    let client = state.mcp_client();

    if !client.is_running() {
        client.spawn().await.map_err(|e| e.to_string())?;
    }

    let n = limit.unwrap_or(50);

    let args = serde_json::json!({
        "dataset": dataset_name,
        "n": n
    });

    let result = client
        .call_tool("head_dataset", args)
        .await
        .map_err(|e| e.to_string())?;

    if !result.success {
        return Err(result.error.unwrap_or_else(|| "Unknown error".to_string()));
    }

    parse_head_result(&result.content, &dataset_name, offset.unwrap_or(0), n)
}

/// Describe dataset statistics.
#[tauri::command]
pub async fn describe_dataset(
    state: State<'_, AppState>,
    dataset_name: String,
) -> Result<String, String> {
    let client = state.mcp_client();

    if !client.is_running() {
        client.spawn().await.map_err(|e| e.to_string())?;
    }

    let args = serde_json::json!({
        "dataset": dataset_name
    });

    let result = client
        .call_tool("describe_dataset", args)
        .await
        .map_err(|e| e.to_string())?;

    if !result.success {
        return Err(result.error.unwrap_or_else(|| "Unknown error".to_string()));
    }

    Ok(result.content)
}

// Helper functions to parse tool output

fn parse_dataset_list(content: &str) -> Result<Vec<DatasetInfo>, String> {
    // The list_datasets tool returns formatted text
    // For now, return empty if no datasets
    if content.contains("No datasets") {
        return Ok(vec![]);
    }

    // Parse the formatted output
    // This is a simplified parser - in production, you'd want more robust parsing
    let mut datasets = Vec::new();

    for line in content.lines() {
        // Look for lines like "  1. dataset_name (100 rows x 5 columns)"
        if let Some(start) = line.find(". ") {
            let rest = &line[start + 2..];
            if let Some(paren_start) = rest.find(" (") {
                let name = rest[..paren_start].trim().to_string();
                let dims = &rest[paren_start + 2..];

                // Parse dimensions
                if let Some(x_pos) = dims.find(" x ") {
                    if let Ok(rows) = dims[..x_pos].trim().replace(" rows", "").parse::<usize>() {
                        if let Some(cols_end) = dims.find(" columns") {
                            if let Ok(cols) = dims[x_pos + 3..cols_end].trim().parse::<usize>() {
                                datasets.push(DatasetInfo {
                                    name,
                                    rows,
                                    columns: cols,
                                    column_names: vec![],
                                    source_path: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(datasets)
}

fn parse_load_result(content: &str, path: &str) -> Result<DatasetInfo, String> {
    // Parse output like:
    // Successfully loaded dataset 'sample_csv'
    // Dimensions: 1000 rows x 5 columns
    // Columns: id (i64), name (str), value (f64), ...

    let mut name = String::new();
    let mut rows = 0;
    let mut columns = 0;
    let mut column_names = Vec::new();

    for line in content.lines() {
        if line.starts_with("Successfully loaded dataset") {
            if let Some(start) = line.find('\'') {
                if let Some(end) = line[start + 1..].find('\'') {
                    name = line[start + 1..start + 1 + end].to_string();
                }
            }
        } else if line.starts_with("Dimensions:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            for (i, part) in parts.iter().enumerate() {
                if *part == "rows" && i > 0 {
                    rows = parts[i - 1].parse().unwrap_or(0);
                }
                if *part == "columns" && i > 0 {
                    columns = parts[i - 1].parse().unwrap_or(0);
                }
            }
        } else if line.starts_with("Columns:") {
            let cols_part = &line[8..];
            for col in cols_part.split(',') {
                let col = col.trim();
                if let Some(paren) = col.find('(') {
                    column_names.push(col[..paren].trim().to_string());
                } else {
                    column_names.push(col.to_string());
                }
            }
        }
    }

    if name.is_empty() {
        return Err("Failed to parse load result".to_string());
    }

    Ok(DatasetInfo {
        name,
        rows,
        columns,
        column_names,
        source_path: Some(path.to_string()),
    })
}

fn parse_head_result(
    content: &str,
    name: &str,
    offset: usize,
    limit: usize,
) -> Result<DatasetPreview, String> {
    // The head_dataset tool returns a formatted table
    // This is a simplified parser

    let lines: Vec<&str> = content.lines().collect();
    let mut columns = Vec::new();
    let mut rows = Vec::new();
    let mut total_rows = 0;

    // Look for header line (usually after the title)
    let mut in_table = false;
    for line in lines {
        if line.contains("rows x") {
            // Extract total rows from dimension line
            if let Some(pos) = line.find(" rows") {
                let start = line[..pos].rfind(' ').unwrap_or(0);
                total_rows = line[start..pos].trim().parse().unwrap_or(0);
            }
        } else if line.starts_with('│') && line.contains('│') {
            let parts: Vec<&str> = line
                .split('│')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim())
                .collect();

            if !in_table {
                // First row is headers
                columns = parts.iter().map(|s| s.to_string()).collect();
                in_table = true;
            } else if !parts.is_empty() && !parts[0].contains("───") {
                // Data row
                let row: Vec<serde_json::Value> = parts
                    .iter()
                    .map(|s| {
                        // Try to parse as number
                        if let Ok(n) = s.parse::<f64>() {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(n)
                                    .unwrap_or_else(|| serde_json::Number::from(0)),
                            )
                        } else if let Ok(n) = s.parse::<i64>() {
                            serde_json::Value::Number(serde_json::Number::from(n))
                        } else {
                            serde_json::Value::String(s.to_string())
                        }
                    })
                    .collect();
                if !row.is_empty() {
                    rows.push(row);
                }
            }
        }
    }

    Ok(DatasetPreview {
        name: name.to_string(),
        columns,
        rows,
        total_rows,
        offset,
        limit,
    })
}
