//! Data quality and cleaning tools handlers.
//!
//! This module provides MCP tool handlers for data quality profiling and cleaning:
//! - Data quality profile generation
//! - Cleaning operation preview and verification
//! - Cleaning session management (start, status, apply, rollback)
//! - Smart cleaning suggestions

use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, model::*, tool, tool_router,
};

use p2a_core::data::munging::clean::fill_na_with;
use p2a_core::data::{
    CleaningOperation, CleaningSession, SuggestionPriority, deduplicate, drop_na, filter,
    generate_quality_profile, generate_suggestions, preview_cleaning, replace, to_lowercase,
    to_uppercase, trim, verify_cleaning,
};

use crate::server::AnalyticsServer;
use crate::tools::handlers::munging::{DROP_HOW, fill_label, parse_choice, parse_fill_strategy};
use crate::tools::requests::cleaning::{
    CleaningRollbackRequest, CleaningSessionApplyRequest, CleaningSessionCheckpointsRequest,
    CleaningSessionStartRequest, CleaningSessionStatusRequest, DataQualityProfileRequest,
    ListCleaningSessionsRequest, PreviewCleaningRequest, SuggestCleaningRequest,
    VerifyCleaningRequest,
};
use p2a_core::options::Choices;

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

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
        description = "Preview a data cleaning operation before applying it. Shows how many rows would be affected, sample changes, and warnings. Supports: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter. fill_na strategies: mean, median, mode, forward, backward, zero, constant (with value); a value without a strategy means constant."
    )]
    pub async fn preview_cleaning(
        &self,
        Parameters(request): Parameters<PreviewCleaningRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = crate::get_dataset!(datasets, &request.dataset);

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
            "fill_na" | "fillna" => {
                let strategy = match parse_fill_strategy(
                    request.strategy.as_deref(),
                    request.value.as_deref(),
                ) {
                    Ok(strategy) => strategy,
                    Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
                };
                CleaningOperation::FillNa {
                    columns: request.columns,
                    strategy: strategy.kind().to_string(),
                    value: request.value,
                }
            }
            "drop_na" | "dropna" => {
                let how = match parse_choice(&DROP_HOW, request.how.as_deref().unwrap_or("any")) {
                    Ok(how) => how,
                    Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
                };
                CleaningOperation::DropNa {
                    columns: request.columns,
                    how: how.to_string(),
                }
            }
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

        let before = crate::get_dataset!(datasets, &request.before_dataset);

        let after = crate::get_dataset!(datasets, &request.after_dataset);

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
        sessions.insert(session_id.clone(), Arc::new(session));

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
        description = "Apply a cleaning operation within a session. Creates a new checkpoint automatically. Returns a verification report showing what changed. fill_na strategies: mean, median, mode, forward, backward, zero, constant (with value); a value without a strategy means constant. Pass output_dataset to store the session's current data as a dataset that analysis tools can read."
    )]
    pub async fn cleaning_session_apply(
        &self,
        Parameters(request): Parameters<CleaningSessionApplyRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut sessions = self.cleaning_sessions.write().await;
        let Some(shared) = sessions.get_mut(&request.session_id) else {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                request.session_id
            ))]));
        };

        let operation_type = request.operation.to_lowercase();
        let params = std::collections::HashMap::new();

        // Helper to convert Vec<String> to Vec<&str>
        fn to_str_slice(v: &Option<Vec<String>>) -> Option<Vec<&str>> {
            v.as_ref()
                .map(|cols| cols.iter().map(|s| s.as_str()).collect())
        }
        let first_column = |operation: &str| {
            request
                .columns
                .as_ref()
                .and_then(|c| c.first())
                .cloned()
                .ok_or_else(|| {
                    McpError::invalid_request(format!("{operation} requires a column"), None)
                })
        };
        type Apply<'a> = Box<
            dyn FnOnce(&p2a_core::data::Dataset) -> Result<p2a_core::data::Dataset, String>
                + Send
                + 'a,
        >;
        // What a fill wrote, for the reply (the session's report counts cells, not values).
        let mut fill_note = String::new();

        // The operation and its required parameters are resolved before the
        // session is touched, so a request that is refused copies nothing.
        let (description, apply): (String, Apply<'_>) = match operation_type.as_str() {
            "trim" => {
                let cols = request.columns.clone();
                (
                    format!("{} operation", operation_type),
                    Box::new(move |ds| {
                        trim(ds, to_str_slice(&cols).as_deref()).map_err(|e| e.to_string())
                    }),
                )
            }
            "lowercase" | "to_lowercase" => {
                let column = first_column("lowercase")?;
                (
                    format!("lowercase {}", column),
                    Box::new(move |ds| to_lowercase(ds, &column).map_err(|e| e.to_string())),
                )
            }
            "uppercase" | "to_uppercase" => {
                let column = first_column("uppercase")?;
                (
                    format!("uppercase {}", column),
                    Box::new(move |ds| to_uppercase(ds, &column).map_err(|e| e.to_string())),
                )
            }
            "fill_na" | "fillna" => {
                let strategy = match parse_fill_strategy(
                    request.strategy.as_deref(),
                    request.value.as_deref(),
                ) {
                    Ok(strategy) => strategy,
                    Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
                };
                let cols = request.columns.clone();
                let note = &mut fill_note;
                (
                    format!("fill_na ({})", fill_label(&strategy)),
                    Box::new(move |ds| {
                        let outcome = fill_na_with(ds, to_str_slice(&cols).as_deref(), &strategy)
                            .map_err(|e| e.to_string())?;
                        *note = outcome.summary();
                        Ok(outcome.dataset)
                    }),
                )
            }
            "drop_na" | "dropna" => {
                let how = match parse_choice(&DROP_HOW, request.how.as_deref().unwrap_or("any")) {
                    Ok(how) => how,
                    Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
                };
                let cols = request.columns.clone();
                (
                    format!("drop_na ({})", how),
                    Box::new(move |ds| {
                        drop_na(ds, to_str_slice(&cols).as_deref(), how).map_err(|e| e.to_string())
                    }),
                )
            }
            "deduplicate" | "dedup" => {
                let keep = request.keep.as_deref().unwrap_or("first").to_string();
                let cols = request.columns.clone();
                (
                    format!("deduplicate (keep={})", keep),
                    Box::new(move |ds| {
                        deduplicate(ds, to_str_slice(&cols).as_deref(), &keep)
                            .map_err(|e| e.to_string())
                    }),
                )
            }
            "replace" => {
                let column = first_column("replace")?;
                let old_value = request
                    .old_value
                    .clone()
                    .ok_or_else(|| McpError::invalid_request("replace requires old_value", None))?;
                let new_value = request
                    .value
                    .clone()
                    .ok_or_else(|| McpError::invalid_request("replace requires value", None))?;
                (
                    format!("replace '{}' with '{}' in {}", old_value, new_value, column),
                    Box::new(move |ds| {
                        replace(ds, &column, &old_value, &new_value).map_err(|e| e.to_string())
                    }),
                )
            }
            "filter" => {
                let column = first_column("filter")?;
                let operator = request
                    .operator
                    .clone()
                    .ok_or_else(|| McpError::invalid_request("filter requires operator", None))?;
                let value = request.filter_value.clone().ok_or_else(|| {
                    McpError::invalid_request("filter requires filter_value", None)
                })?;
                (
                    format!("filter {} {} {}", column, operator, value),
                    Box::new(move |ds| {
                        filter(ds, &column, &operator, &value).map_err(|e| e.to_string())
                    }),
                )
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown operation '{}'. Supported: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter",
                    request.operation
                ))]));
            }
        };

        // Copied only now, when a quota snapshot shares it, so that the
        // snapshot keeps the state before this call.
        let result =
            Arc::make_mut(shared).apply_operation(&operation_type, &description, params, apply);

        match result {
            Ok(report) => {
                let mut reply = String::from("Operation applied successfully!\n\n");
                if !fill_note.is_empty() {
                    reply.push_str(&fill_note);
                    reply.push_str("\n\n");
                }
                reply.push_str(&report.summary());
                let current = request
                    .output_dataset
                    .is_some()
                    .then(|| shared.current_dataset().clone());
                drop(sessions);
                reply.push_str(
                    &self
                        .store_session_result(
                            "cleaning_session_apply",
                            request.output_dataset,
                            current,
                        )
                        .await?,
                );
                Ok(CallToolResult::success(vec![Content::text(reply)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Operation failed: {}",
                e
            ))])),
        }
    }

    /// Rollback a cleaning session to a previous checkpoint.
    #[tool(
        description = "Rollback a cleaning session to a previous checkpoint. If no checkpoint index is provided, rolls back to the previous checkpoint (undo last operation). Pass output_dataset to store the session's current data as a dataset that analysis tools can read."
    )]
    pub async fn cleaning_rollback(
        &self,
        Parameters(request): Parameters<CleaningRollbackRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut sessions = self.cleaning_sessions.write().await;
        let Some(shared) = sessions.get_mut(&request.session_id) else {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                request.session_id
            ))]));
        };

        // A rollback the session refuses is refused on the shared session,
        // so it copies nothing; one that runs copies it first when a quota
        // snapshot shares it (see `cleaning_session_apply`).
        if let Some(e) = rollback_refusal(shared, request.checkpoint_index) {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "Rollback failed: {}",
                e
            ))]));
        }
        let session = Arc::make_mut(shared);

        let result = match request.checkpoint_index {
            Some(index) => session.rollback_to(index),
            None => session.rollback(),
        };

        match result {
            Ok(()) => {
                let status = session.status();
                let mut reply = format!(
                    "Rollback successful!\n\n\
                     Current checkpoint: {}\n\
                     Rows: {}\n\
                     Completeness: {:.1}%",
                    status.current_checkpoint,
                    status.current_row_count,
                    status.current_completeness * 100.0
                );
                let current = request
                    .output_dataset
                    .is_some()
                    .then(|| session.current_dataset().clone());
                drop(sessions);
                reply.push_str(
                    &self
                        .store_session_result("cleaning_rollback", request.output_dataset, current)
                        .await?,
                );
                Ok(CallToolResult::success(vec![Content::text(reply)]))
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
        let min_priority = match request.min_priority.as_deref() {
            None => None,
            Some(p) => match parse_choice(&MIN_PRIORITY, p) {
                Ok(p) => Some(p),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            },
        };

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
                    suggested_value(s.operation.as_str(), &s.parameters),
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

/// The `min_priority` of `suggest_cleaning`.
const MIN_PRIORITY: Choices<SuggestionPriority> = Choices {
    option: "min_priority",
    values: &[
        (&["low"], SuggestionPriority::Low),
        (&["medium"], SuggestionPriority::Medium),
        (&["high"], SuggestionPriority::High),
        (&["critical"], SuggestionPriority::Critical),
    ],
};

/// The `- value:` of a suggestion. A computed fill value (the median or mode `suggest_cleaning`
/// shows) is for reference only: `fill_na` computes it itself and refuses a value with any strategy
/// but 'constant'.
fn suggested_value(operation: &str, parameters: &p2a_core::data::SuggestionParameters) -> String {
    let value = parameters.value.as_deref().unwrap_or("-");
    match (
        operation,
        parameters.value.as_deref(),
        parameters.strategy.as_deref(),
    ) {
        ("fill_na", Some(_), Some(strategy))
            if !strategy.trim().eq_ignore_ascii_case("constant") =>
        {
            format!("{value} (shown for reference; pass only strategy '{strategy}' to fill_na)")
        }
        _ => value.to_string(),
    }
}

impl AnalyticsServer {
    /// The end of a cleaning step's reply: stores `current` (the session's
    /// dataset after the step, cloned with the session lock released) under
    /// `output_dataset` and says so, or says how to keep the result. With
    /// quotas on, the stored copy is charged as a dataset first; a shortfall
    /// is the quota refusal, and the step stays applied in the session.
    async fn store_session_result(
        &self,
        tool: &str,
        output_dataset: Option<String>,
        current: Option<p2a_core::data::Dataset>,
    ) -> Result<String, McpError> {
        let (Some(name), Some(dataset)) = (output_dataset, current) else {
            return Ok(
                "\nThe result is in the cleaning session only; pass output_dataset to \
                 store it as a dataset."
                    .to_string(),
            );
        };
        if crate::session::dataset_admission_active() {
            let bytes: u64 = dataset
                .df()
                .get_columns()
                .iter()
                .map(|c| crate::session::series_bytes(c.as_materialized_series()))
                .fold(0, u64::saturating_add);
            if let Err(shortfall) = crate::session::admit_charge(&name, bytes).await {
                return Err(crate::session::quota_refusal(format!(
                    "{tool} applied its step in the cleaning session, but storing the result as \
                     '{name}' would take {}, so {}. The step stays applied \
                     in the cleaning session; nothing was stored.",
                    crate::session::fmt_bytes(bytes),
                    shortfall.outcome()
                )));
            }
        }
        let (rows, cols) = (dataset.nrows(), dataset.ncols());
        self.datasets.write().await.insert(name.clone(), dataset);
        Ok(format!(
            "\nSaved as '{name}' ({rows} rows x {cols} columns)."
        ))
    }
}

/// The error `CleaningSession::rollback_to` (`Some(index)`) or `rollback`
/// (`None`) would return for this request, without changing the session:
/// an index past the last checkpoint, or no index at the initial state.
fn rollback_refusal(session: &CleaningSession, index: Option<usize>) -> Option<String> {
    match index {
        Some(index) if session.get_checkpoint_dataset(index).is_none() => Some(format!(
            "Invalid checkpoint index {}. Valid range: 0-{}",
            index,
            session.list_checkpoints().len().saturating_sub(1)
        )),
        None if session.current_checkpoint == 0 => {
            Some("Already at initial state, cannot rollback further".to_string())
        }
        _ => None,
    }
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use std::sync::Arc;

    use p2a_core::data::CleaningSession;
    use serde_json::json;

    use crate::server::AnalyticsServer;
    use crate::session::Session;
    use crate::session::test_fixtures::cleaning_fixture;

    /// A session holding one cleaning session `c1` at its initial checkpoint.
    async fn session_with_cleaning() -> Session {
        let s = Session::new("cleaning".into(), None);
        s.cleaning_sessions.write().await.insert(
            "c1".into(),
            Arc::new(CleaningSession::new(cleaning_fixture(), "cl")),
        );
        s
    }

    /// Whether the session still holds the snapshot's `c1`, not a copy.
    async fn shares_c1(s: &Session, snapshot: &crate::session::SessionStateSnapshot) -> bool {
        Arc::ptr_eq(
            &s.cleaning_sessions.read().await["c1"],
            &snapshot.cleaning_sessions["c1"],
        )
    }

    /// The refusal text of a failed call, from its error or its content.
    fn refusal(r: &crate::transport::http::ToolResult) -> String {
        assert!(!r.success);
        let content: Vec<&str> = r
            .content
            .iter()
            .filter_map(|c| match c {
                crate::transport::http::ContentItem::Text { text } => Some(text.as_str()),
                crate::transport::http::ContentItem::Image { .. } => None,
            })
            .collect();
        r.error.clone().unwrap_or_else(|| content.join("\n"))
    }

    /// A refused rollback or apply leaves the cleaning session shared with a
    /// quota snapshot: pre-fix the handler copied it (`Arc::make_mut`)
    /// before it found that the request changes nothing.
    #[tokio::test]
    async fn refused_cleaning_requests_copy_nothing() {
        let s = session_with_cleaning().await;
        let snapshot = s.snapshot_state().await;
        let server = AnalyticsServer::new();
        let call = |tool: &'static str, args: serde_json::Value| {
            let (server, s) = (&server, &s);
            async move { server.call_tool_with_session(tool, args, s).await }
        };

        // No earlier checkpoint to roll back to, with and without an index.
        let r = call("cleaning_rollback", json!({"session_id": "c1"}))
            .await
            .unwrap();
        let expected = CleaningSession::new(cleaning_fixture(), "cl")
            .rollback()
            .unwrap_err();
        assert_eq!(refusal(&r), format!("Rollback failed: {expected}"));
        assert!(shares_c1(&s, &snapshot).await, "refused rollback");
        let r = call(
            "cleaning_rollback",
            json!({"session_id": "c1", "checkpoint_index": 9}),
        )
        .await
        .unwrap();
        let expected = CleaningSession::new(cleaning_fixture(), "cl")
            .rollback_to(9)
            .unwrap_err();
        assert_eq!(refusal(&r), format!("Rollback failed: {expected}"));
        assert!(shares_c1(&s, &snapshot).await, "refused rollback_to");

        // An unknown operation, and one without its required parameters.
        let r = call(
            "cleaning_session_apply",
            json!({"session_id": "c1", "operation": "shuffle"}),
        )
        .await
        .unwrap();
        assert!(refusal(&r).contains("Unknown operation 'shuffle'"), "{r:?}");
        assert!(shares_c1(&s, &snapshot).await, "unknown operation");
        let missing = call(
            "cleaning_session_apply",
            json!({"session_id": "c1", "operation": "replace", "columns": ["v"], "value": "0"}),
        )
        .await;
        assert!(
            format!("{missing:?}").contains("replace requires old_value"),
            "{missing:?}"
        );
        assert!(shares_c1(&s, &snapshot).await, "missing parameter");

        // An operation that runs changes the session, which is then copied.
        let r = call(
            "cleaning_session_apply",
            json!({"session_id": "c1", "operation": "drop_na"}),
        )
        .await
        .unwrap();
        assert!(r.success, "{r:?}");
        assert!(!shares_c1(&s, &snapshot).await, "applied operation");
    }
}
