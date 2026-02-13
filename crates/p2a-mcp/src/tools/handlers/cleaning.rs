//! Data quality and cleaning tools handlers.
//!
//! This module provides MCP tool handlers for data quality profiling and cleaning:
//! - Data quality profile generation
//! - Cleaning operation preview and verification
//! - Cleaning session management (start, status, apply, rollback)
//! - Smart cleaning suggestions

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use p2a_core::data::{
    CleaningOperation, CleaningSession, FillStrategy, SuggestionPriority, deduplicate, drop_na,
    fill_na, filter, generate_quality_profile, generate_suggestions, preview_cleaning, replace,
    to_lowercase, to_uppercase, trim, verify_cleaning,
};

use crate::server::AnalyticsServer;
use crate::tools::requests::cleaning::{
    CleaningRollbackRequest, CleaningSessionApplyRequest, CleaningSessionCheckpointsRequest,
    CleaningSessionStartRequest, CleaningSessionStatusRequest, DataQualityProfileRequest,
    ListCleaningSessionsRequest, PreviewCleaningRequest, SuggestCleaningRequest,
    VerifyCleaningRequest,
};

#[tool_router(router = cleaning_router, vis = "pub")]
impl AnalyticsServer {
    /// Generate a comprehensive data quality profile.
    #[tool(
        description = "Generate a comprehensive data quality profile for LLM-assisted data cleaning. Returns column-level statistics (nulls, uniques, types), numeric outlier detection, string pattern analysis, and automated issue detection with severity ratings."
    )]
    pub async fn data_quality_profile(
        &self,
        Parameters(request): Parameters<DataQualityProfileRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let profile = generate_quality_profile(dataset);

        // Format the profile for LLM consumption
        let mut result = profile.summary();

        // Add detailed column information
        result.push_str("\n\nColumn Details:\n");
        result.push_str("===============\n");

        for col in &profile.columns {
            result.push_str(&format!("\n## {} ({})\n", col.name, col.dtype));
            result.push_str(&format!(
                "  - Null: {} ({:.1}%)\n",
                col.null_count,
                col.null_pct * 100.0
            ));
            result.push_str(&format!(
                "  - Unique: {} ({:.1}%)\n",
                col.unique_count,
                col.unique_pct * 100.0
            ));

            if let Some(ref stats) = col.numeric_stats {
                result.push_str(&format!(
                    "  - Range: {:.2} to {:.2}\n",
                    stats.min, stats.max
                ));
                result.push_str(&format!(
                    "  - Mean: {:.2}, Median: {:.2}, Std: {:.2}\n",
                    stats.mean, stats.median, stats.std
                ));
                if stats.outlier_count > 0 {
                    result.push_str(&format!(
                        "  - Outliers: {} (outside {:.2} to {:.2})\n",
                        stats.outlier_count, stats.outlier_lower_bound, stats.outlier_upper_bound
                    ));
                }
            }

            if let Some(ref stats) = col.string_stats {
                result.push_str(&format!(
                    "  - Length: {} to {} chars (avg {:.1})\n",
                    stats.min_length, stats.max_length, stats.mean_length
                ));
                if !stats.detected_patterns.is_empty() {
                    result.push_str(&format!(
                        "  - Patterns: {}\n",
                        stats.detected_patterns.join(", ")
                    ));
                }
                if !stats.top_values.is_empty() {
                    let top_3: Vec<String> = stats
                        .top_values
                        .iter()
                        .take(3)
                        .map(|(v, c)| format!("'{}' ({})", v, c))
                        .collect();
                    result.push_str(&format!("  - Top values: {}\n", top_3.join(", ")));
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Preview what a cleaning operation would do before applying it.
    #[tool(
        description = "Preview a data cleaning operation before applying it. Shows how many rows would be affected, sample changes, and warnings. Supports: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter."
    )]
    pub async fn preview_cleaning(
        &self,
        Parameters(request): Parameters<PreviewCleaningRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Build the CleaningOperation from request parameters
        let operation = match request.operation.to_lowercase().as_str() {
            "trim" => CleaningOperation::Trim {
                columns: request.columns,
            },
            "lowercase" | "to_lowercase" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| {
                        McpError::invalid_request("lowercase operation requires a column", None)
                    })?
                    .clone();
                CleaningOperation::ToLowercase { column }
            }
            "uppercase" | "to_uppercase" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| {
                        McpError::invalid_request("uppercase operation requires a column", None)
                    })?
                    .clone();
                CleaningOperation::ToUppercase { column }
            }
            "fill_na" | "fillna" => CleaningOperation::FillNa {
                columns: request.columns,
                strategy: request.strategy.unwrap_or_else(|| "constant".to_string()),
                value: request.value,
            },
            "drop_na" | "dropna" => CleaningOperation::DropNa {
                columns: request.columns,
                how: request.how.unwrap_or_else(|| "any".to_string()),
            },
            "deduplicate" | "dedup" => CleaningOperation::Deduplicate {
                columns: request.columns,
                keep: request.keep.unwrap_or_else(|| "first".to_string()),
            },
            "replace" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| {
                        McpError::invalid_request("replace operation requires a column", None)
                    })?
                    .clone();
                let old_value = request.old_value.ok_or_else(|| {
                    McpError::invalid_request("replace operation requires old_value", None)
                })?;
                let new_value = request.value.ok_or_else(|| {
                    McpError::invalid_request("replace operation requires value (new value)", None)
                })?;
                CleaningOperation::Replace {
                    column,
                    old_value,
                    new_value,
                }
            }
            "filter" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| {
                        McpError::invalid_request("filter operation requires a column", None)
                    })?
                    .clone();
                let operator = request.operator.ok_or_else(|| {
                    McpError::invalid_request("filter operation requires operator", None)
                })?;
                let value = request.filter_value.ok_or_else(|| {
                    McpError::invalid_request("filter operation requires filter_value", None)
                })?;
                CleaningOperation::Filter {
                    column,
                    operator,
                    value,
                }
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown operation '{}'. Supported: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter",
                    request.operation
                ))]));
            }
        };

        let sample_size = request.sample_size.unwrap_or(5);
        let preview = preview_cleaning(dataset, &operation, sample_size);

        Ok(CallToolResult::success(vec![Content::text(
            preview.summary(),
        )]))
    }

    /// Verify a cleaning operation by comparing before and after datasets.
    #[tool(
        description = "Verify a cleaning operation by comparing the original and cleaned datasets. Returns a detailed report with row counts, quality delta (completeness change, issues resolved/introduced), and sample changes."
    )]
    pub async fn verify_cleaning(
        &self,
        Parameters(request): Parameters<VerifyCleaningRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let before = match datasets.get(&request.before_dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.before_dataset
                ))]));
            }
        };

        let after = match datasets.get(&request.after_dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.after_dataset
                ))]));
            }
        };

        let report = verify_cleaning(before, after, &request.operation_description);

        Ok(CallToolResult::success(vec![Content::text(
            report.summary(),
        )]))
    }

    /// Start a new cleaning session for a dataset.
    #[tool(
        description = "Start a new cleaning session for a dataset. Returns a session ID that can be used to track progress, apply operations, and rollback changes. Each session maintains checkpoints for undo capability."
    )]
    pub async fn cleaning_session_start(
        &self,
        Parameters(request): Parameters<CleaningSessionStartRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds.clone(),
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };
        drop(datasets);

        let session_name = request
            .session_name
            .unwrap_or_else(|| request.dataset.clone());
        let session = CleaningSession::new(dataset, &session_name);
        let session_id = session.id.clone();
        let status = session.status();

        let mut sessions = self.cleaning_sessions.write().await;
        sessions.insert(session_id.clone(), session);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Cleaning session started!\n\n\
             Session ID: {}\n\
             Dataset: {}\n\
             Rows: {}\n\
             Completeness: {:.1}%\n\n\
             Use 'cleaning_session_apply' to apply cleaning operations.\n\
             Use 'cleaning_session_status' to check progress.\n\
             Use 'cleaning_rollback' to undo operations.",
            session_id,
            session_name,
            status.current_row_count,
            status.current_completeness * 100.0
        ))]))
    }

    /// Get the status of a cleaning session.
    #[tool(
        description = "Get the current status of a cleaning session, including checkpoint count, operations performed, current row count, and completeness score."
    )]
    pub async fn cleaning_session_status(
        &self,
        Parameters(request): Parameters<CleaningSessionStatusRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sessions = self.cleaning_sessions.read().await;

        let session = match sessions.get(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(
            session.summary(),
        )]))
    }

    /// List all active cleaning sessions.
    #[tool(description = "List all active cleaning sessions with their current status.")]
    pub async fn list_cleaning_sessions(
        &self,
        Parameters(_request): Parameters<ListCleaningSessionsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sessions = self.cleaning_sessions.read().await;

        if sessions.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No active cleaning sessions.\n\nUse 'cleaning_session_start' to start a new session.",
            )]));
        }

        let mut result = String::from("Active Cleaning Sessions\n");
        result.push_str("========================\n\n");

        for (id, session) in sessions.iter() {
            let status = session.status();
            result.push_str(&format!(
                "Session: {} ({})\n  - Checkpoints: {}\n  - Operations: {}\n  - Rows: {}\n  - Completeness: {:.1}%\n\n",
                id, status.dataset_name, status.total_checkpoints, status.total_operations,
                status.current_row_count, status.current_completeness * 100.0
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Apply a cleaning operation within a session.
    #[tool(
        description = "Apply a cleaning operation within a session. Creates a new checkpoint automatically. Returns a verification report showing what changed."
    )]
    pub async fn cleaning_session_apply(
        &self,
        Parameters(request): Parameters<CleaningSessionApplyRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut sessions = self.cleaning_sessions.write().await;

        let session = match sessions.get_mut(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        let operation_type = request.operation.to_lowercase();
        let description = format!("{} operation", operation_type);
        let params = std::collections::HashMap::new();

        // Helper to convert Vec<String> to Vec<&str>
        fn to_str_slice(v: &Option<Vec<String>>) -> Option<Vec<&str>> {
            v.as_ref()
                .map(|cols| cols.iter().map(|s| s.as_str()).collect())
        }

        // Apply the operation based on type
        let result = match operation_type.as_str() {
            "trim" => {
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                session.apply_operation(&operation_type, &description, params, move |ds| {
                    trim(ds, cols_ref.as_deref()).map_err(|e| e.to_string())
                })
            }
            "lowercase" | "to_lowercase" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("lowercase requires a column", None))?
                    .clone();
                session.apply_operation(
                    &operation_type,
                    &format!("lowercase {}", column),
                    params,
                    |ds| to_lowercase(ds, &column).map_err(|e| e.to_string()),
                )
            }
            "uppercase" | "to_uppercase" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("uppercase requires a column", None))?
                    .clone();
                session.apply_operation(
                    &operation_type,
                    &format!("uppercase {}", column),
                    params,
                    |ds| to_uppercase(ds, &column).map_err(|e| e.to_string()),
                )
            }
            "fill_na" | "fillna" => {
                let strategy_str = request.strategy.as_deref().unwrap_or("constant");
                let value = request.value.clone();
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                let strategy = match strategy_str {
                    "mean" => FillStrategy::Mean,
                    "median" => FillStrategy::Median,
                    "forward" => FillStrategy::Forward,
                    "backward" => FillStrategy::Backward,
                    "constant" | _ => {
                        FillStrategy::Constant(value.unwrap_or_else(|| "0".to_string()))
                    }
                };
                session.apply_operation(
                    &operation_type,
                    &format!("fill_na with {:?}", strategy),
                    params,
                    move |ds| {
                        fill_na(ds, cols_ref.as_deref(), strategy.clone())
                            .map_err(|e| e.to_string())
                    },
                )
            }
            "drop_na" | "dropna" => {
                let how = request.how.as_deref().unwrap_or("any").to_string();
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                session.apply_operation(
                    &operation_type,
                    &format!("drop_na ({})", how),
                    params,
                    move |ds| drop_na(ds, cols_ref.as_deref(), &how).map_err(|e| e.to_string()),
                )
            }
            "deduplicate" | "dedup" => {
                let keep = request.keep.as_deref().unwrap_or("first").to_string();
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                session.apply_operation(
                    &operation_type,
                    &format!("deduplicate (keep={})", keep),
                    params,
                    move |ds| {
                        deduplicate(ds, cols_ref.as_deref(), &keep).map_err(|e| e.to_string())
                    },
                )
            }
            "replace" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("replace requires a column", None))?
                    .clone();
                let old_value = request
                    .old_value
                    .clone()
                    .ok_or_else(|| McpError::invalid_request("replace requires old_value", None))?;
                let new_value = request
                    .value
                    .clone()
                    .ok_or_else(|| McpError::invalid_request("replace requires value", None))?;
                session.apply_operation(
                    &operation_type,
                    &format!("replace '{}' with '{}' in {}", old_value, new_value, column),
                    params,
                    |ds| replace(ds, &column, &old_value, &new_value).map_err(|e| e.to_string()),
                )
            }
            "filter" => {
                let column = request
                    .columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("filter requires a column", None))?
                    .clone();
                let operator = request
                    .operator
                    .clone()
                    .ok_or_else(|| McpError::invalid_request("filter requires operator", None))?;
                let value = request.filter_value.clone().ok_or_else(|| {
                    McpError::invalid_request("filter requires filter_value", None)
                })?;
                session.apply_operation(
                    &operation_type,
                    &format!("filter {} {} {}", column, operator, value),
                    params,
                    |ds| filter(ds, &column, &operator, &value).map_err(|e| e.to_string()),
                )
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown operation '{}'. Supported: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter",
                    request.operation
                ))]));
            }
        };

        match result {
            Ok(report) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Operation applied successfully!\n\n{}",
                report.summary()
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Operation failed: {}",
                e
            ))])),
        }
    }

    /// Rollback a cleaning session to a previous checkpoint.
    #[tool(
        description = "Rollback a cleaning session to a previous checkpoint. If no checkpoint index is provided, rolls back to the previous checkpoint (undo last operation)."
    )]
    pub async fn cleaning_rollback(
        &self,
        Parameters(request): Parameters<CleaningRollbackRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut sessions = self.cleaning_sessions.write().await;

        let session = match sessions.get_mut(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        let result = match request.checkpoint_index {
            Some(index) => session.rollback_to(index),
            None => session.rollback(),
        };

        match result {
            Ok(()) => {
                let status = session.status();
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Rollback successful!\n\n\
                     Current checkpoint: {}\n\
                     Rows: {}\n\
                     Completeness: {:.1}%",
                    status.current_checkpoint,
                    status.current_row_count,
                    status.current_completeness * 100.0
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Rollback failed: {}",
                e
            ))])),
        }
    }

    /// List all checkpoints in a cleaning session.
    #[tool(
        description = "List all checkpoints in a cleaning session, showing the state at each point."
    )]
    pub async fn cleaning_session_checkpoints(
        &self,
        Parameters(request): Parameters<CleaningSessionCheckpointsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sessions = self.cleaning_sessions.read().await;

        let session = match sessions.get(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        let checkpoints = session.list_checkpoints();
        let mut result = String::from("Session Checkpoints\n");
        result.push_str("===================\n\n");

        for cp in checkpoints {
            let marker = if cp.is_current { " <-- current" } else { "" };
            result.push_str(&format!(
                "#{}: {}{}\n  - Rows: {}\n  - Completeness: {:.1}%\n  - Created: {}\n\n",
                cp.index,
                cp.description,
                marker,
                cp.row_count,
                cp.completeness * 100.0,
                cp.created_at.format("%H:%M:%S")
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Generate smart cleaning suggestions for a dataset.
    #[tool(
        description = "Analyze a dataset and generate prioritized cleaning suggestions. Returns specific operations with parameters, estimated impact, and reasoning. Use this to get intelligent recommendations before starting a cleaning workflow."
    )]
    pub async fn suggest_cleaning(
        &self,
        Parameters(request): Parameters<SuggestCleaningRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'load_dataset' first.",
                    request.dataset
                ))]));
            }
        };

        // Generate quality profile and suggestions
        let profile = generate_quality_profile(dataset);
        let report = generate_suggestions(&profile);

        // Parse minimum priority filter
        let min_priority =
            request
                .min_priority
                .as_ref()
                .and_then(|p| match p.to_lowercase().as_str() {
                    "low" => Some(SuggestionPriority::Low),
                    "medium" => Some(SuggestionPriority::Medium),
                    "high" => Some(SuggestionPriority::High),
                    "critical" => Some(SuggestionPriority::Critical),
                    _ => None,
                });

        // Filter suggestions
        let mut suggestions: Vec<_> = report
            .suggestions
            .iter()
            .filter(|s| min_priority.is_none_or(|min| s.priority >= min))
            .collect();

        // Apply limit if specified
        if let Some(limit) = request.limit {
            suggestions.truncate(limit);
        }

        // Build result
        let mut result = format!(
            "Cleaning Suggestions for '{}'\n\
             ================================\n\n\
             Dataset: {} rows x {} columns\n\
             Completeness: {:.1}%\n\
             Issues found: {}\n\
             Suggestions: {}\n\n",
            request.dataset,
            report.dataset_summary.row_count,
            report.dataset_summary.column_count,
            report.dataset_summary.completeness_score * 100.0,
            report.issues_analyzed,
            suggestions.len()
        );

        if suggestions.is_empty() {
            result.push_str("No cleaning suggestions - your data looks clean!\n");
        } else {
            for (i, s) in suggestions.iter().enumerate() {
                result.push_str(&format!(
                    "{}. [{}] {}\n\
                     ----------------------------------------\n\
                     Category: {:?}\n\
                     Issue: {}\n\
                     \n\
                     Description: {}\n\
                     \n\
                     Reasoning: {}\n\
                     \n\
                     Impact: {}\n\
                     \n\
                     Operation: '{}'\n\
                     Parameters:\n\
                     - column: {}\n\
                     - value: {}\n\
                     - strategy: {}\n\
                     \n\
                     Considerations:\n",
                    i + 1,
                    s.priority.label(),
                    s.title,
                    s.category,
                    s.addresses_issue,
                    s.description,
                    s.reasoning,
                    s.estimated_impact.impact_description,
                    s.operation,
                    s.parameters.column.as_deref().unwrap_or("-"),
                    s.parameters.value.as_deref().unwrap_or("-"),
                    s.parameters.strategy.as_deref().unwrap_or("-"),
                ));

                for consideration in &s.considerations {
                    result.push_str(&format!("  - {}\n", consideration));
                }
                result.push('\n');
            }

            // Add overall recommendation
            result.push_str(&format!(
                "\nRecommendation\n--------------\n{}\n",
                report.overall_recommendation
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}
