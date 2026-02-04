//! Utility tool handlers (seed management, reports, session export/import).
//!
//! This module defines utility tools using the `#[tool_router(router = utils_router)]` pattern.

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::utils::{
    ColumnSpecInput, ExportSessionRequest, GenerateRandomDataRequest, GenerateReportRequest,
    GetSeedRequest, ImportSessionRequest, ReportContentInput, ReportSectionInput,
    ServerStatsRequest, SetSeedRequest,
};

use p2a_core::{
    HtmlReport, ReportSection, ReportTable,
    simulation::{ColumnSpec, Distribution, generate_random_data},
};

#[tool_router(router = utils_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Seed Management Tools
    // ========================================================================

    /// Set the global random seed for ML reproducibility.
    #[tool(
        description = "Set a global random seed for ML operations (kmeans, random_forest, tsne). When set, ML tools will use this seed as a fallback if no per-tool seed is specified. Clear by calling with no seed value."
    )]
    pub async fn set_seed(
        &self,
        Parameters(request): Parameters<SetSeedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut global_seed = self.global_seed.write().await;
        *global_seed = request.seed;

        match request.seed {
            Some(seed) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Global random seed set to: {}\n\
                 This seed will be used by ML tools (kmeans, random_forest, tsne) unless overridden per-tool.",
                seed
            ))])),
            None => Ok(CallToolResult::success(vec![Content::text(
                "Global random seed cleared. ML tools will use random initialization unless a per-tool seed is specified.".to_string()
            )])),
        }
    }

    /// Get the current global random seed.
    #[tool(
        description = "Get the current global random seed setting and list which ML tools support seeded reproducibility."
    )]
    pub async fn get_seed(
        &self,
        Parameters(_request): Parameters<GetSeedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let global_seed = self.global_seed.read().await;

        let seed_status = match *global_seed {
            Some(seed) => format!("Current global seed: {}", seed),
            None => "No global seed set (using random initialization)".to_string(),
        };

        let output = format!(
            "Seed Management\n{}\n\
             {}\n\n\
             ML tools supporting reproducibility:\n\
             - ml_kmeans: Uses seed for centroid initialization\n\
             - ml_random_forest: Uses seed for bootstrap sampling and feature selection\n\
             - ml_tsne: Uses seed for initial embedding\n\n\
             Per-tool seeds override the global seed.",
            "=".repeat(40),
            seed_status
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Server Stats Tools
    // ========================================================================

    /// Get server memory and performance statistics.
    #[tool(
        description = "Get memory usage statistics for the analytics server. Shows dataset memory consumption, process memory, and memory trends. Useful for monitoring resource usage with large datasets."
    )]
    pub async fn server_stats(
        &self,
        Parameters(request): Parameters<ServerStatsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::{format_bytes, get_process_memory};

        let include_datasets = request.include_datasets.unwrap_or(true);
        let include_history = request.include_history.unwrap_or(false);

        let datasets = self.datasets.read().await;
        let profiler = self.memory_profiler.read().await;

        // Get detailed stats with dataset dimensions
        let stats = profiler.stats_with_datasets(datasets.iter().map(|(k, v)| (k.as_str(), v)));

        let mut output = String::new();
        output.push_str("Server Memory Statistics\n");
        output.push_str(&"=".repeat(50));
        output.push_str("\n\n");

        // Summary section
        output.push_str("## Summary\n\n");
        output.push_str(&format!(
            "Total Dataset Memory: {}\n",
            stats.dataset_memory_formatted
        ));
        output.push_str(&format!("Datasets Loaded: {}\n", stats.dataset_count));
        output.push_str(&format!(
            "Peak Memory: {}\n",
            format_bytes(profiler.peak_memory())
        ));
        output.push_str(&format!(
            "Uptime: {:.1}s\n",
            profiler.elapsed().as_secs_f64()
        ));

        // Process memory (if available)
        if let Some(ref proc_mem) = stats.process_memory {
            output.push_str(&format!("\nProcess RSS: {}\n", proc_mem.rss_formatted));
            output.push_str(&format!(
                "Process Virtual: {}\n",
                proc_mem.virtual_formatted
            ));
        } else if let Some(proc_mem) = get_process_memory() {
            output.push_str(&format!("\nProcess RSS: {}\n", proc_mem.rss_formatted));
            output.push_str(&format!(
                "Process Virtual: {}\n",
                proc_mem.virtual_formatted
            ));
        }

        // Per-dataset breakdown
        if include_datasets && !stats.datasets.is_empty() {
            output.push_str("\n## Datasets (by memory usage)\n\n");
            output.push_str(&format!(
                "{:<30} {:>12} {:>10} {:>10}\n",
                "Name", "Memory", "Rows", "Cols"
            ));
            output.push_str(&"-".repeat(64));
            output.push('\n');

            for ds in &stats.datasets {
                output.push_str(&format!(
                    "{:<30} {:>12} {:>10} {:>10}\n",
                    if ds.name.len() > 28 {
                        format!("{}...", &ds.name[..25])
                    } else {
                        ds.name.clone()
                    },
                    ds.memory_formatted,
                    ds.rows,
                    ds.columns
                ));
            }
        }

        // Memory history
        if include_history {
            let history = profiler.history();
            if !history.is_empty() {
                output.push_str("\n## Recent Memory History\n\n");
                output.push_str(&format!(
                    "{:>12} {:>15} {:>10}\n",
                    "Time (ms)", "Memory", "Datasets"
                ));
                output.push_str(&"-".repeat(40));
                output.push('\n');

                // Show last 10 entries
                let start = if history.len() > 10 {
                    history.len() - 10
                } else {
                    0
                };
                for snapshot in &history[start..] {
                    output.push_str(&format!(
                        "{:>12} {:>15} {:>10}\n",
                        snapshot.time_offset_ms,
                        format_bytes(snapshot.total_bytes),
                        snapshot.dataset_count
                    ));
                }
            }
        }

        drop(datasets);
        drop(profiler);

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Random Data Generation Tools
    // ========================================================================

    /// Generate random data with specified distributions.
    #[tool(
        description = "Generate a random dataset with specified columns and distributions. Supports: uniform (min, max), normal (mean, std), binomial (n, p), poisson (lambda), exponential (rate), bernoulli (p), categorical (categories, optional weights), uniform_int (min, max), sequence (start), constant (value), constant_string (value). Example column: {\"name\": \"x\", \"distribution\": {\"type\": \"normal\", \"mean\": 0, \"std\": 1}}"
    )]
    pub async fn generate_random_data(
        &self,
        Parameters(request): Parameters<GenerateRandomDataRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Parse column specifications
        let mut columns: Vec<ColumnSpec> = Vec::with_capacity(request.columns.len());

        for col_input in &request.columns {
            let dist: Distribution = match serde_json::from_value(col_input.distribution.clone()) {
                Ok(d) => d,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Invalid distribution specification for column '{}': {}\n\n\
                         Expected format: {{\"type\": \"distribution_name\", ...params}}\n\
                         Available types:\n\
                         - uniform: {{\"type\": \"uniform\", \"min\": 0.0, \"max\": 1.0}}\n\
                         - normal: {{\"type\": \"normal\", \"mean\": 0.0, \"std\": 1.0}}\n\
                         - binomial: {{\"type\": \"binomial\", \"n\": 10, \"p\": 0.5}}\n\
                         - poisson: {{\"type\": \"poisson\", \"lambda\": 5.0}}\n\
                         - exponential: {{\"type\": \"exponential\", \"rate\": 1.0}}\n\
                         - bernoulli: {{\"type\": \"bernoulli\", \"p\": 0.5}}\n\
                         - categorical: {{\"type\": \"categorical\", \"categories\": [\"A\", \"B\", \"C\"], \"weights\": [0.5, 0.3, 0.2]}}\n\
                         - uniform_int: {{\"type\": \"uniform_int\", \"min\": 1, \"max\": 10}}\n\
                         - sequence: {{\"type\": \"sequence\", \"start\": 1}}\n\
                         - constant: {{\"type\": \"constant\", \"value\": 42.0}}\n\
                         - constant_string: {{\"type\": \"constant_string\", \"value\": \"text\"}}",
                        col_input.name, e
                    ))]));
                }
            };

            columns.push(ColumnSpec::new(&col_input.name, dist));
        }

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        // Generate the data
        let dataset = match generate_random_data(request.n_rows, columns, seed) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to generate random data: {}",
                    e
                ))]));
            }
        };

        // Get dataset info
        let n_rows = dataset.df().height();
        let n_cols = dataset.df().width();
        let col_names: Vec<String> = dataset
            .df()
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        // Determine dataset name
        let name = request.name.unwrap_or_else(|| "generated".to_string());

        // Store the dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "Random Dataset Generated\n{}\n\
             Name: {}\n\
             Rows: {}\n\
             Columns: {}\n\
             Column names: {}\n\
             Seed: {}\n\n\
             The dataset '{}' is now available for analysis.",
            "=".repeat(40),
            name,
            n_rows,
            n_cols,
            col_names.join(", "),
            seed.map(|s| s.to_string())
                .unwrap_or_else(|| "random".to_string()),
            name
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Report Generation Tools
    // ========================================================================

    /// Generate an HTML report from structured analysis results.
    #[tool(
        description = "Generate a self-contained HTML report from analysis results. The report includes proper styling, tables, charts (as embedded images), and is suitable for sharing or printing. Returns the complete HTML document as a string."
    )]
    pub async fn generate_report(
        &self,
        Parameters(request): Parameters<GenerateReportRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Build the report structure
        let mut report = HtmlReport::new(&request.title);

        if let Some(ref subtitle) = request.subtitle {
            report = report.with_subtitle(subtitle);
        }

        if let Some(ref author) = request.author {
            report = report.with_author(author);
        }

        // Process each section
        for section_input in &request.sections {
            let mut section = ReportSection::new(&section_input.title);

            for content_input in &section_input.content {
                match content_input.content_type.as_str() {
                    "text" => {
                        if let Some(ref text) = content_input.text {
                            section.add_text(text);
                        }
                    }
                    "code" => {
                        if let Some(ref code) = content_input.text {
                            section.add_code(code, content_input.language.as_deref());
                        }
                    }
                    "table" => {
                        if let (Some(headers), Some(rows)) =
                            (&content_input.headers, &content_input.rows)
                        {
                            let mut table = ReportTable::new(headers.clone());
                            if let Some(ref caption) = content_input.caption {
                                table = table.with_caption(caption);
                            }
                            for row in rows {
                                table.add_row(row.clone());
                            }
                            section.add_table(table);
                        }
                    }
                    "chart" => {
                        if let Some(ref image) = content_input.image_base64 {
                            section.add_chart(
                                image,
                                content_input.chart_title.as_deref(),
                                content_input.chart_caption.as_deref(),
                            );
                        }
                    }
                    "stats" => {
                        if let Some(ref stats) = content_input.stats {
                            let items: Vec<(String, String)> = stats
                                .iter()
                                .filter_map(|pair| {
                                    if pair.len() >= 2 {
                                        Some((pair[0].clone(), pair[1].clone()))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            section.add_statistics(items);
                        }
                    }
                    _ => {
                        // Unknown content type, skip
                    }
                }
            }

            report.add_section(section);
        }

        // Generate the HTML
        let html = report.to_html();

        // Return the HTML - it's quite long so we provide summary info
        let summary = format!(
            "HTML Report Generated\n\
             =====================\n\
             Title: {}\n\
             Sections: {}\n\
             HTML Length: {} characters\n\n\
             The complete HTML report follows:\n\n{}",
            request.title,
            request.sections.len(),
            html.len(),
            html
        );

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }

    // ========================================================================
    // Session Export/Import Tools
    // ========================================================================

    /// Export the current analysis session to a JSON file.
    #[tool(
        description = "Export the current session including all loaded datasets and their metadata. Can save to file or return as string. Useful for saving your analysis state to resume later."
    )]
    pub async fn export_session(
        &self,
        Parameters(request): Parameters<ExportSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;
        use std::fs;

        let datasets = self.datasets.read().await;
        let include_data = request.include_data.unwrap_or(true);

        let mut session_data = serde_json::Map::new();
        session_data.insert("version".to_string(), serde_json::json!("1.0"));
        session_data.insert(
            "created_at".to_string(),
            serde_json::json!(chrono::Utc::now().to_rfc3339()),
        );

        let mut datasets_json = serde_json::Map::new();
        for (name, dataset) in datasets.iter() {
            let df = dataset.df();
            let mut ds_info = serde_json::Map::new();

            // Save schema
            let schema: Vec<serde_json::Value> = df
                .get_columns()
                .iter()
                .map(|col| {
                    serde_json::json!({
                        "name": col.name().to_string(),
                        "dtype": format!("{:?}", col.dtype())
                    })
                })
                .collect();
            ds_info.insert("schema".to_string(), serde_json::json!(schema));
            ds_info.insert("n_rows".to_string(), serde_json::json!(df.height()));
            ds_info.insert("n_cols".to_string(), serde_json::json!(df.width()));

            if include_data {
                // Serialize actual data
                let mut columns_data = serde_json::Map::new();
                for col in df.get_columns() {
                    let col_name = col.name().to_string();
                    let values: Vec<serde_json::Value> = (0..col.len())
                        .map(|i| match col.get(i) {
                            Ok(av) => match av {
                                AnyValue::Null => serde_json::Value::Null,
                                AnyValue::Boolean(b) => serde_json::json!(b),
                                AnyValue::Int8(v) => serde_json::json!(v),
                                AnyValue::Int16(v) => serde_json::json!(v),
                                AnyValue::Int32(v) => serde_json::json!(v),
                                AnyValue::Int64(v) => serde_json::json!(v),
                                AnyValue::UInt8(v) => serde_json::json!(v),
                                AnyValue::UInt16(v) => serde_json::json!(v),
                                AnyValue::UInt32(v) => serde_json::json!(v),
                                AnyValue::UInt64(v) => serde_json::json!(v),
                                AnyValue::Float32(v) => serde_json::json!(v),
                                AnyValue::Float64(v) => serde_json::json!(v),
                                AnyValue::String(s) => serde_json::json!(s),
                                _ => serde_json::json!(format!("{:?}", av)),
                            },
                            Err(_) => serde_json::Value::Null,
                        })
                        .collect();
                    columns_data.insert(col_name, serde_json::json!(values));
                }
                ds_info.insert("data".to_string(), serde_json::json!(columns_data));
            }

            datasets_json.insert(name.clone(), serde_json::json!(ds_info));
        }
        session_data.insert("datasets".to_string(), serde_json::json!(datasets_json));

        let json_output = serde_json::to_string_pretty(&session_data).map_err(|e| {
            McpError::internal_error(format!("JSON serialization failed: {}", e), None)
        })?;

        if let Some(file_path) = request.file_path {
            fs::write(&file_path, &json_output).map_err(|e| {
                McpError::internal_error(format!("Failed to write session file: {}", e), None)
            })?;

            Ok(CallToolResult::success(vec![Content::text(format!(
                "Session exported successfully to: {}\n\
                 Datasets saved: {}\n\
                 Include data: {}",
                file_path,
                datasets.len(),
                include_data
            ))]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(format!(
                "Session Export\n{}\n\
                 Datasets: {}\n\n{}",
                "=".repeat(40),
                datasets.len(),
                json_output
            ))]))
        }
    }

    /// Import a previously exported analysis session.
    #[tool(
        description = "Import a previously exported session from a JSON file. Can merge with existing session or replace it. Restores all datasets with their original names."
    )]
    pub async fn import_session(
        &self,
        Parameters(request): Parameters<ImportSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;
        use std::fs;

        let json_content = fs::read_to_string(&request.file_path).map_err(|e| {
            McpError::internal_error(format!("Failed to read session file: {}", e), None)
        })?;

        let session: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| McpError::internal_error(format!("Invalid JSON: {}", e), None))?;

        let datasets_obj = session
            .get("datasets")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                McpError::internal_error("Invalid session format: missing 'datasets' field", None)
            })?;

        let merge = request.merge.unwrap_or(false);
        let mut datasets = self.datasets.write().await;

        if !merge {
            datasets.clear();
        }

        let mut imported_count = 0;
        let mut errors = Vec::new();

        for (name, ds_info) in datasets_obj {
            let ds_obj = match ds_info.as_object() {
                Some(obj) => obj,
                None => {
                    errors.push(format!("{}: invalid format", name));
                    continue;
                }
            };

            // Check if we have data to restore
            if let Some(data) = ds_obj.get("data").and_then(|v| v.as_object()) {
                // Reconstruct DataFrame from stored columns
                let mut columns_vec: Vec<Column> = Vec::new();

                for (col_name, values) in data {
                    if let Some(arr) = values.as_array() {
                        // Try to determine column type from first non-null value
                        let first_non_null = arr.iter().find(|v| !v.is_null());

                        let series: Series = match first_non_null {
                            Some(serde_json::Value::Number(n)) if n.is_f64() => {
                                let vals: Vec<Option<f64>> =
                                    arr.iter().map(|v| v.as_f64()).collect();
                                Series::new(col_name.into(), vals)
                            }
                            Some(serde_json::Value::Number(_)) => {
                                let vals: Vec<Option<i64>> =
                                    arr.iter().map(|v| v.as_i64()).collect();
                                Series::new(col_name.into(), vals)
                            }
                            Some(serde_json::Value::Bool(_)) => {
                                let vals: Vec<Option<bool>> =
                                    arr.iter().map(|v| v.as_bool()).collect();
                                Series::new(col_name.into(), vals)
                            }
                            _ => {
                                // Default to string
                                let vals: Vec<Option<String>> = arr
                                    .iter()
                                    .map(|v| {
                                        if v.is_null() {
                                            None
                                        } else if let Some(s) = v.as_str() {
                                            Some(s.to_string())
                                        } else {
                                            Some(v.to_string())
                                        }
                                    })
                                    .collect();
                                Series::new(col_name.into(), vals)
                            }
                        };
                        columns_vec.push(series.into());
                    }
                }

                if !columns_vec.is_empty() {
                    match DataFrame::new(columns_vec) {
                        Ok(df) => {
                            let dataset = p2a_core::Dataset::new(df);
                            datasets.insert(name.clone(), dataset);
                            imported_count += 1;
                        }
                        Err(e) => {
                            errors.push(format!("{}: DataFrame error - {}", name, e));
                        }
                    }
                } else {
                    errors.push(format!("{}: no column data found", name));
                }
            } else {
                errors.push(format!("{}: no data field (metadata-only session)", name));
            }
        }

        let mut output = format!(
            "Session Import\n{}\n\
             File: {}\n\
             Mode: {}\n\
             Datasets imported: {}\n",
            "=".repeat(40),
            request.file_path,
            if merge { "merge" } else { "replace" },
            imported_count
        );

        if !errors.is_empty() {
            output.push_str(&format!("\nErrors:\n{}", errors.join("\n")));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }
}
