//! Cleaning session management for LLM-assisted data cleaning workflows.
//!
//! This module provides:
//! - Session state management for multi-step cleaning workflows
//! - Rollback points with dataset snapshots
//! - Audit trail of all operations
//! - Session persistence for long workflows

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::Dataset;
use super::quality::{DataQualityProfile, generate_quality_profile};
use super::verification::{VerificationReport, verify_cleaning};

/// A cleaning session that tracks the state of a data cleaning workflow.
#[derive(Clone)]
pub struct CleaningSession {
    /// Unique session identifier
    pub id: String,
    /// Name of the dataset being cleaned
    pub dataset_name: String,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Last activity time
    pub updated_at: DateTime<Utc>,
    /// Current checkpoint index (0 = initial state)
    pub current_checkpoint: usize,
    /// All checkpoints in order
    checkpoints: Vec<SessionCheckpoint>,
    /// Audit trail of all operations
    pub audit_trail: Vec<OperationRecord>,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

/// A checkpoint representing the dataset state at a point in time.
#[derive(Clone)]
pub struct SessionCheckpoint {
    /// Unique checkpoint identifier
    pub id: String,
    /// Checkpoint index (sequential)
    pub index: usize,
    /// When this checkpoint was created
    pub created_at: DateTime<Utc>,
    /// Description of what led to this checkpoint
    pub description: String,
    /// The dataset state at this checkpoint
    dataset: Dataset,
    /// Quality profile at this checkpoint
    pub quality_profile: DataQualityProfile,
}

/// Record of an operation performed during the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationRecord {
    /// Unique operation identifier
    pub id: String,
    /// When the operation was performed
    pub timestamp: DateTime<Utc>,
    /// Type of operation (e.g., "trim", "fill_na", "filter")
    pub operation_type: String,
    /// Human-readable description
    pub description: String,
    /// Parameters used for the operation
    pub parameters: HashMap<String, String>,
    /// Checkpoint index before the operation
    pub checkpoint_before: usize,
    /// Checkpoint index after the operation (if successful)
    pub checkpoint_after: Option<usize>,
    /// Whether the operation was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Verification report (if operation was verified)
    pub verification: Option<VerificationReportSummary>,
}

/// Serializable summary of a verification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReportSummary {
    pub rows_before: usize,
    pub rows_after: usize,
    pub rows_modified: usize,
    pub rows_removed: usize,
    pub completeness_before: f64,
    pub completeness_after: f64,
    pub issues_resolved: usize,
    pub issues_introduced: usize,
}

impl From<&VerificationReport> for VerificationReportSummary {
    fn from(report: &VerificationReport) -> Self {
        VerificationReportSummary {
            rows_before: report.rows_before,
            rows_after: report.rows_after,
            rows_modified: report.rows_modified,
            rows_removed: report.rows_removed,
            completeness_before: report.quality_delta.completeness_before,
            completeness_after: report.quality_delta.completeness_after,
            issues_resolved: report.quality_delta.issues_resolved.len(),
            issues_introduced: report.quality_delta.issues_introduced.len(),
        }
    }
}

/// Session status information (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatus {
    pub id: String,
    pub dataset_name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub current_checkpoint: usize,
    pub total_checkpoints: usize,
    pub total_operations: usize,
    pub successful_operations: usize,
    pub current_row_count: usize,
    pub current_completeness: f64,
    pub can_rollback: bool,
}

impl CleaningSession {
    /// Create a new cleaning session for a dataset.
    pub fn new(dataset: Dataset, dataset_name: &str) -> Self {
        let now = Utc::now();
        let session_id = Uuid::new_v4().to_string();

        // Create initial checkpoint
        let quality_profile = generate_quality_profile(&dataset);
        let initial_checkpoint = SessionCheckpoint {
            id: Uuid::new_v4().to_string(),
            index: 0,
            created_at: now,
            description: "Initial state".to_string(),
            dataset,
            quality_profile,
        };

        CleaningSession {
            id: session_id,
            dataset_name: dataset_name.to_string(),
            created_at: now,
            updated_at: now,
            current_checkpoint: 0,
            checkpoints: vec![initial_checkpoint],
            audit_trail: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the current dataset state.
    pub fn current_dataset(&self) -> &Dataset {
        &self.checkpoints[self.current_checkpoint].dataset
    }

    /// Get the current quality profile.
    pub fn current_quality_profile(&self) -> &DataQualityProfile {
        &self.checkpoints[self.current_checkpoint].quality_profile
    }

    /// Get the initial dataset state.
    pub fn initial_dataset(&self) -> &Dataset {
        &self.checkpoints[0].dataset
    }

    /// Get session status.
    pub fn status(&self) -> SessionStatus {
        let current = &self.checkpoints[self.current_checkpoint];
        let successful_ops = self.audit_trail.iter().filter(|op| op.success).count();

        SessionStatus {
            id: self.id.clone(),
            dataset_name: self.dataset_name.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            current_checkpoint: self.current_checkpoint,
            total_checkpoints: self.checkpoints.len(),
            total_operations: self.audit_trail.len(),
            successful_operations: successful_ops,
            current_row_count: current.dataset.nrows(),
            current_completeness: current.quality_profile.completeness_score,
            can_rollback: self.current_checkpoint > 0,
        }
    }

    /// Apply a cleaning operation and create a new checkpoint.
    ///
    /// The `apply_fn` should take the current dataset and return the cleaned dataset.
    pub fn apply_operation<F>(
        &mut self,
        operation_type: &str,
        description: &str,
        parameters: HashMap<String, String>,
        apply_fn: F,
    ) -> Result<VerificationReport, String>
    where
        F: FnOnce(&Dataset) -> Result<Dataset, String>,
    {
        let now = Utc::now();
        let operation_id = Uuid::new_v4().to_string();
        let checkpoint_before = self.current_checkpoint;

        // Get the current dataset
        let current_dataset = self.current_dataset();

        // Apply the operation
        match apply_fn(current_dataset) {
            Ok(new_dataset) => {
                // Generate verification report
                let verification = verify_cleaning(current_dataset, &new_dataset, description);

                // Create new checkpoint
                let quality_profile = generate_quality_profile(&new_dataset);
                let new_checkpoint_index = self.checkpoints.len();
                let new_checkpoint = SessionCheckpoint {
                    id: Uuid::new_v4().to_string(),
                    index: new_checkpoint_index,
                    created_at: now,
                    description: description.to_string(),
                    dataset: new_dataset,
                    quality_profile,
                };

                self.checkpoints.push(new_checkpoint);
                self.current_checkpoint = new_checkpoint_index;
                self.updated_at = now;

                // Record the operation
                let record = OperationRecord {
                    id: operation_id,
                    timestamp: now,
                    operation_type: operation_type.to_string(),
                    description: description.to_string(),
                    parameters,
                    checkpoint_before,
                    checkpoint_after: Some(new_checkpoint_index),
                    success: true,
                    error: None,
                    verification: Some(VerificationReportSummary::from(&verification)),
                };
                self.audit_trail.push(record);

                Ok(verification)
            }
            Err(e) => {
                // Record the failed operation
                let record = OperationRecord {
                    id: operation_id,
                    timestamp: now,
                    operation_type: operation_type.to_string(),
                    description: description.to_string(),
                    parameters,
                    checkpoint_before,
                    checkpoint_after: None,
                    success: false,
                    error: Some(e.clone()),
                    verification: None,
                };
                self.audit_trail.push(record);
                self.updated_at = now;

                Err(e)
            }
        }
    }

    /// Rollback to a specific checkpoint.
    pub fn rollback_to(&mut self, checkpoint_index: usize) -> Result<(), String> {
        if checkpoint_index >= self.checkpoints.len() {
            return Err(format!(
                "Invalid checkpoint index {}. Valid range: 0-{}",
                checkpoint_index,
                self.checkpoints.len() - 1
            ));
        }

        let now = Utc::now();

        // Record the rollback operation
        let record = OperationRecord {
            id: Uuid::new_v4().to_string(),
            timestamp: now,
            operation_type: "rollback".to_string(),
            description: format!(
                "Rolled back from checkpoint {} to checkpoint {}",
                self.current_checkpoint, checkpoint_index
            ),
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "from_checkpoint".to_string(),
                    self.current_checkpoint.to_string(),
                );
                params.insert("to_checkpoint".to_string(), checkpoint_index.to_string());
                params
            },
            checkpoint_before: self.current_checkpoint,
            checkpoint_after: Some(checkpoint_index),
            success: true,
            error: None,
            verification: None,
        };
        self.audit_trail.push(record);

        self.current_checkpoint = checkpoint_index;
        self.updated_at = now;

        Ok(())
    }

    /// Rollback to the previous checkpoint (undo last operation).
    pub fn rollback(&mut self) -> Result<(), String> {
        if self.current_checkpoint == 0 {
            return Err("Already at initial state, cannot rollback further".to_string());
        }
        self.rollback_to(self.current_checkpoint - 1)
    }

    /// Get a list of all checkpoints.
    pub fn list_checkpoints(&self) -> Vec<CheckpointInfo> {
        self.checkpoints
            .iter()
            .map(|cp| CheckpointInfo {
                id: cp.id.clone(),
                index: cp.index,
                created_at: cp.created_at,
                description: cp.description.clone(),
                row_count: cp.dataset.nrows(),
                completeness: cp.quality_profile.completeness_score,
                is_current: cp.index == self.current_checkpoint,
            })
            .collect()
    }

    /// Get the audit trail.
    pub fn get_audit_trail(&self) -> &[OperationRecord] {
        &self.audit_trail
    }

    /// Get a specific checkpoint's dataset.
    pub fn get_checkpoint_dataset(&self, index: usize) -> Option<&Dataset> {
        self.checkpoints.get(index).map(|cp| &cp.dataset)
    }

    /// Compare two checkpoints.
    pub fn compare_checkpoints(
        &self,
        index1: usize,
        index2: usize,
    ) -> Result<VerificationReport, String> {
        let cp1 = self
            .checkpoints
            .get(index1)
            .ok_or_else(|| format!("Checkpoint {} not found", index1))?;
        let cp2 = self
            .checkpoints
            .get(index2)
            .ok_or_else(|| format!("Checkpoint {} not found", index2))?;

        Ok(verify_cleaning(
            &cp1.dataset,
            &cp2.dataset,
            "checkpoint comparison",
        ))
    }

    /// Set session metadata.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get session metadata.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Generate a summary of the entire session.
    pub fn summary(&self) -> String {
        let status = self.status();
        let initial = &self.checkpoints[0];
        let current = &self.checkpoints[self.current_checkpoint];

        let mut summary = String::new();
        summary.push_str("Cleaning Session Summary\n");
        summary.push_str("========================\n\n");

        summary.push_str(&format!("Session ID: {}\n", self.id));
        summary.push_str(&format!("Dataset: {}\n", self.dataset_name));
        summary.push_str(&format!(
            "Created: {}\n",
            self.created_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        summary.push_str(&format!(
            "Last updated: {}\n\n",
            self.updated_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));

        summary.push_str("Progress:\n");
        summary.push_str(&format!(
            "  - Checkpoints: {} (current: #{})\n",
            status.total_checkpoints, status.current_checkpoint
        ));
        summary.push_str(&format!(
            "  - Operations: {} ({} successful)\n",
            status.total_operations, status.successful_operations
        ));
        summary.push_str(&format!(
            "  - Can rollback: {}\n\n",
            if status.can_rollback { "Yes" } else { "No" }
        ));

        summary.push_str("Data Changes:\n");
        summary.push_str(&format!(
            "  - Rows: {} → {}",
            initial.dataset.nrows(),
            current.dataset.nrows()
        ));
        let row_diff = current.dataset.nrows() as i64 - initial.dataset.nrows() as i64;
        if row_diff != 0 {
            summary.push_str(&format!(" ({:+})", row_diff));
        }
        summary.push('\n');

        summary.push_str(&format!(
            "  - Completeness: {:.1}% → {:.1}% ({:+.1}%)\n",
            initial.quality_profile.completeness_score * 100.0,
            current.quality_profile.completeness_score * 100.0,
            (current.quality_profile.completeness_score
                - initial.quality_profile.completeness_score)
                * 100.0
        ));

        let initial_issues: usize = initial
            .quality_profile
            .columns
            .iter()
            .map(|c| c.issues.len())
            .sum();
        let current_issues: usize = current
            .quality_profile
            .columns
            .iter()
            .map(|c| c.issues.len())
            .sum();
        summary.push_str(&format!(
            "  - Issues: {} → {}\n",
            initial_issues, current_issues
        ));

        if !self.audit_trail.is_empty() {
            summary.push_str("\nRecent Operations:\n");
            for (i, op) in self.audit_trail.iter().rev().take(5).enumerate() {
                let status_icon = if op.success { "✓" } else { "✗" };
                summary.push_str(&format!(
                    "  {}. [{}] {} - {}\n",
                    self.audit_trail.len() - i,
                    status_icon,
                    op.operation_type,
                    op.description
                ));
            }
            if self.audit_trail.len() > 5 {
                summary.push_str(&format!(
                    "  ... and {} more operations\n",
                    self.audit_trail.len() - 5
                ));
            }
        }

        summary
    }
}

/// Information about a checkpoint (serializable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub id: String,
    pub index: usize,
    pub created_at: DateTime<Utc>,
    pub description: String,
    pub row_count: usize,
    pub completeness: f64,
    pub is_current: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{IntoLazy, PolarsError, col, df};

    fn create_test_dataset() -> Dataset {
        let test_df = df! {
            "name" => ["  Alice  ", "Bob", " Charlie ", "David"],
            "email" => ["alice@test.com", "bob@test.com ", "charlie@test.com", "david@test.com"],
            "value" => [Some(100), Some(200), None, Some(400)],
        }
        .unwrap();
        Dataset::new(test_df)
    }

    #[test]
    fn test_session_creation() {
        let dataset = create_test_dataset();
        let session = CleaningSession::new(dataset, "test_data");

        assert_eq!(session.dataset_name, "test_data");
        assert_eq!(session.current_checkpoint, 0);
        assert_eq!(session.checkpoints.len(), 1);
        assert!(session.audit_trail.is_empty());
    }

    #[test]
    fn test_session_status() {
        let dataset = create_test_dataset();
        let session = CleaningSession::new(dataset, "test_data");
        let status = session.status();

        assert_eq!(status.dataset_name, "test_data");
        assert_eq!(status.current_checkpoint, 0);
        assert_eq!(status.total_checkpoints, 1);
        assert_eq!(status.total_operations, 0);
        assert!(!status.can_rollback);
    }

    #[test]
    fn test_apply_operation() {
        let dataset = create_test_dataset();
        let mut session = CleaningSession::new(dataset, "test_data");

        // Apply a simple operation that removes a row
        let result = session.apply_operation(
            "filter",
            "Remove rows with null values",
            HashMap::new(),
            |ds| {
                let df = ds
                    .df()
                    .clone()
                    .lazy()
                    .filter(col("value").is_not_null())
                    .collect()
                    .map_err(|e: PolarsError| e.to_string())?;
                Ok(Dataset::new(df))
            },
        );

        assert!(result.is_ok());
        assert_eq!(session.current_checkpoint, 1);
        assert_eq!(session.checkpoints.len(), 2);
        assert_eq!(session.audit_trail.len(), 1);
        assert!(session.audit_trail[0].success);
        assert_eq!(session.current_dataset().nrows(), 3); // One row removed
    }

    #[test]
    fn test_rollback() {
        let dataset = create_test_dataset();
        let mut session = CleaningSession::new(dataset, "test_data");

        // Apply an operation
        let _ = session.apply_operation(
            "filter",
            "Remove rows with null values",
            HashMap::new(),
            |ds| {
                let df = ds
                    .df()
                    .clone()
                    .lazy()
                    .filter(col("value").is_not_null())
                    .collect()
                    .map_err(|e: PolarsError| e.to_string())?;
                Ok(Dataset::new(df))
            },
        );

        assert_eq!(session.current_dataset().nrows(), 3);

        // Rollback
        let rollback_result = session.rollback();
        assert!(rollback_result.is_ok());
        assert_eq!(session.current_checkpoint, 0);
        assert_eq!(session.current_dataset().nrows(), 4); // Back to original
    }

    #[test]
    fn test_rollback_at_initial_state() {
        let dataset = create_test_dataset();
        let mut session = CleaningSession::new(dataset, "test_data");

        let result = session.rollback();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot rollback"));
    }

    #[test]
    fn test_list_checkpoints() {
        let dataset = create_test_dataset();
        let mut session = CleaningSession::new(dataset, "test_data");

        // Apply an operation
        let _ =
            session.apply_operation(
                "test",
                "Test operation",
                HashMap::new(),
                |ds| Ok(ds.clone()),
            );

        let checkpoints = session.list_checkpoints();
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[0].description, "Initial state");
        assert_eq!(checkpoints[1].description, "Test operation");
        assert!(!checkpoints[0].is_current);
        assert!(checkpoints[1].is_current);
    }

    #[test]
    fn test_failed_operation() {
        let dataset = create_test_dataset();
        let mut session = CleaningSession::new(dataset, "test_data");

        let result = session.apply_operation("invalid", "This will fail", HashMap::new(), |_| {
            Err("Intentional failure".to_string())
        });

        assert!(result.is_err());
        assert_eq!(session.current_checkpoint, 0); // Should not have changed
        assert_eq!(session.checkpoints.len(), 1); // No new checkpoint
        assert_eq!(session.audit_trail.len(), 1); // But operation is recorded
        assert!(!session.audit_trail[0].success);
        assert!(session.audit_trail[0].error.is_some());
    }

    #[test]
    fn test_session_summary() {
        let dataset = create_test_dataset();
        let session = CleaningSession::new(dataset, "test_data");
        let summary = session.summary();

        assert!(summary.contains("Cleaning Session Summary"));
        assert!(summary.contains("test_data"));
        assert!(summary.contains("Checkpoints: 1"));
    }

    #[test]
    fn test_compare_checkpoints() {
        let dataset = create_test_dataset();
        let mut session = CleaningSession::new(dataset, "test_data");

        // Apply an operation that changes the data
        let _ = session.apply_operation("filter", "Remove nulls", HashMap::new(), |ds| {
            let df = ds
                .df()
                .clone()
                .lazy()
                .filter(col("value").is_not_null())
                .collect()
                .map_err(|e: PolarsError| e.to_string())?;
            Ok(Dataset::new(df))
        });

        let comparison = session.compare_checkpoints(0, 1);
        assert!(comparison.is_ok());
        let report = comparison.unwrap();
        assert_eq!(report.rows_before, 4);
        assert_eq!(report.rows_after, 3);
        assert_eq!(report.rows_removed, 1);
    }

    #[test]
    fn test_metadata() {
        let dataset = create_test_dataset();
        let mut session = CleaningSession::new(dataset, "test_data");

        session.set_metadata("analyst", "John Doe");
        session.set_metadata("project", "Sales Cleanup");

        assert_eq!(
            session.get_metadata("analyst"),
            Some(&"John Doe".to_string())
        );
        assert_eq!(
            session.get_metadata("project"),
            Some(&"Sales Cleanup".to_string())
        );
        assert_eq!(session.get_metadata("nonexistent"), None);
    }
}
