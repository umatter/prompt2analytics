//! Smart cleaning suggestions for LLM-assisted data cleaning.
//!
//! This module analyzes data quality profiles and generates prioritized
//! cleaning suggestions with specific operations and parameters.

use serde::{Deserialize, Serialize};

use super::quality::{ColumnProfile, DataIssue, DataQualityProfile};

/// Priority level for cleaning suggestions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuggestionPriority {
    /// Low priority - minor issues, optional fixes
    Low = 1,
    /// Medium priority - should be addressed
    Medium = 2,
    /// High priority - important to fix
    High = 3,
    /// Critical priority - must fix before analysis
    Critical = 4,
}

impl SuggestionPriority {
    /// Get a human-readable label for the priority.
    pub fn label(&self) -> &'static str {
        match self {
            SuggestionPriority::Low => "Low",
            SuggestionPriority::Medium => "Medium",
            SuggestionPriority::High => "High",
            SuggestionPriority::Critical => "Critical",
        }
    }
}

/// Category of cleaning operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleaningCategory {
    /// Remove or handle missing values
    MissingValues,
    /// Remove duplicate records
    Deduplication,
    /// Fix string formatting issues
    StringFormatting,
    /// Handle outlier values
    OutlierHandling,
    /// Type conversion or standardization
    TypeStandardization,
    /// Data validation
    Validation,
}

impl CleaningCategory {
    /// Get a human-readable label for the category.
    pub fn label(&self) -> &'static str {
        match self {
            CleaningCategory::MissingValues => "Missing Values",
            CleaningCategory::Deduplication => "Deduplication",
            CleaningCategory::StringFormatting => "String Formatting",
            CleaningCategory::OutlierHandling => "Outlier Handling",
            CleaningCategory::TypeStandardization => "Type Standardization",
            CleaningCategory::Validation => "Validation",
        }
    }
}

/// A suggested cleaning operation with all details needed to execute it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleaningSuggestion {
    /// Unique identifier for this suggestion
    pub id: String,
    /// Human-readable title
    pub title: String,
    /// Detailed description of what this operation does
    pub description: String,
    /// The issue this suggestion addresses
    pub addresses_issue: String,
    /// Priority level
    pub priority: SuggestionPriority,
    /// Category of cleaning
    pub category: CleaningCategory,
    /// Target column(s) - None for dataset-level operations
    pub columns: Option<Vec<String>>,
    /// The operation type (matches preview_cleaning/apply_cleaning)
    pub operation: String,
    /// Parameters for the operation
    pub parameters: SuggestionParameters,
    /// Estimated rows that will be affected
    pub estimated_impact: EstimatedImpact,
    /// Reasoning for this suggestion
    pub reasoning: String,
    /// Potential risks or considerations
    pub considerations: Vec<String>,
}

/// Parameters for a suggested cleaning operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionParameters {
    /// Target column name
    pub column: Option<String>,
    /// Operation-specific value (e.g., fill value, filter condition)
    pub value: Option<String>,
    /// Additional columns (e.g., for deduplication subset)
    pub additional_columns: Option<Vec<String>>,
    /// Strategy parameter (e.g., fill strategy)
    pub strategy: Option<String>,
}

/// Estimated impact of applying a suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedImpact {
    /// Number of rows that will be affected
    pub rows_affected: usize,
    /// Percentage of total rows
    pub rows_affected_pct: f64,
    /// Expected change in completeness score
    pub completeness_change: Option<f64>,
    /// Brief description of the impact
    pub impact_description: String,
}

/// Result of generating suggestions from a quality profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionReport {
    /// Generated suggestions, sorted by priority
    pub suggestions: Vec<CleaningSuggestion>,
    /// Total number of issues analyzed
    pub issues_analyzed: usize,
    /// Dataset summary for context
    pub dataset_summary: DatasetSummary,
    /// Overall recommendation
    pub overall_recommendation: String,
}

/// Summary of dataset characteristics for context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSummary {
    /// Total rows
    pub row_count: usize,
    /// Total columns
    pub column_count: usize,
    /// Current completeness score
    pub completeness_score: f64,
    /// Number of issues found
    pub issue_count: usize,
}

/// Generate cleaning suggestions from a data quality profile.
pub fn generate_suggestions(profile: &DataQualityProfile) -> SuggestionReport {
    let mut suggestions = Vec::new();
    let mut suggestion_id = 0;

    // Helper to generate unique IDs
    let mut next_id = || {
        suggestion_id += 1;
        format!("suggestion_{}", suggestion_id)
    };

    // Process dataset-level issues first
    for issue in &profile.dataset_issues {
        if let Some(suggestion) = suggestion_for_dataset_issue(issue, profile, &mut next_id) {
            suggestions.push(suggestion);
        }
    }

    // Process column-level issues
    for col_profile in &profile.columns {
        for issue in &col_profile.issues {
            if let Some(suggestion) =
                suggestion_for_column_issue(issue, col_profile, profile, &mut next_id)
            {
                suggestions.push(suggestion);
            }
        }
    }

    // Sort by priority (highest first), then by estimated impact
    suggestions.sort_by(|a, b| {
        b.priority.cmp(&a.priority).then_with(|| {
            b.estimated_impact
                .rows_affected
                .cmp(&a.estimated_impact.rows_affected)
        })
    });

    // Calculate overall recommendation
    let overall_recommendation = generate_overall_recommendation(&suggestions, profile);

    let issues_analyzed = profile.dataset_issues.len()
        + profile
            .columns
            .iter()
            .map(|c| c.issues.len())
            .sum::<usize>();

    SuggestionReport {
        suggestions,
        issues_analyzed,
        dataset_summary: DatasetSummary {
            row_count: profile.row_count,
            column_count: profile.columns.len(),
            completeness_score: profile.completeness_score,
            issue_count: issues_analyzed,
        },
        overall_recommendation,
    }
}

/// Generate a suggestion for a dataset-level issue.
fn suggestion_for_dataset_issue<F>(
    issue: &DataIssue,
    profile: &DataQualityProfile,
    next_id: &mut F,
) -> Option<CleaningSuggestion>
where
    F: FnMut() -> String,
{
    match issue {
        DataIssue::DuplicateRows { count } => {
            let pct = *count as f64 / profile.row_count as f64 * 100.0;
            Some(CleaningSuggestion {
                id: next_id(),
                title: "Remove duplicate rows".to_string(),
                description: format!(
                    "Remove {} duplicate rows ({:.1}% of dataset) to ensure data integrity.",
                    count, pct
                ),
                addresses_issue: issue.description(),
                priority: if pct > 10.0 { SuggestionPriority::High } else { SuggestionPriority::Medium },
                category: CleaningCategory::Deduplication,
                columns: None,
                operation: "deduplicate".to_string(),
                parameters: SuggestionParameters {
                    column: None,
                    value: None,
                    additional_columns: None,
                    strategy: Some("first".to_string()),
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: *count,
                    rows_affected_pct: pct,
                    completeness_change: None,
                    impact_description: format!("Will remove {} rows", count),
                },
                reasoning: "Duplicate rows can skew analysis results and inflate counts. Removing them ensures each observation is counted once.".to_string(),
                considerations: vec![
                    "Verify duplicates are true duplicates and not valid repeated measurements".to_string(),
                    "Consider which duplicate to keep (first, last, or based on a timestamp)".to_string(),
                ],
            })
        }
        DataIssue::PossibleDuplicates { columns, count } => {
            let pct = *count as f64 / profile.row_count as f64 * 100.0;
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Investigate possible duplicates in {}", columns.join(", ")),
                description: format!(
                    "Found {} rows ({:.1}%) that may be duplicates based on columns: {}",
                    count,
                    pct,
                    columns.join(", ")
                ),
                addresses_issue: issue.description(),
                priority: SuggestionPriority::Low,
                category: CleaningCategory::Deduplication,
                columns: Some(columns.clone()),
                operation: "deduplicate".to_string(),
                parameters: SuggestionParameters {
                    column: None,
                    value: None,
                    additional_columns: Some(columns.clone()),
                    strategy: Some("first".to_string()),
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: *count,
                    rows_affected_pct: pct,
                    completeness_change: None,
                    impact_description: format!("May affect up to {} rows", count),
                },
                reasoning:
                    "These rows share the same values in key columns and may represent duplicates."
                        .to_string(),
                considerations: vec![
                    "Review sample duplicates before removal".to_string(),
                    "These may be valid records with intentionally repeated values".to_string(),
                ],
            })
        }
        _ => None, // Other dataset issues don't have direct suggestions
    }
}

/// Generate a suggestion for a column-level issue.
fn suggestion_for_column_issue<F>(
    issue: &DataIssue,
    col_profile: &ColumnProfile,
    profile: &DataQualityProfile,
    next_id: &mut F,
) -> Option<CleaningSuggestion>
where
    F: FnMut() -> String,
{
    match issue {
        DataIssue::HighNullRate { column, pct } => {
            let null_count = col_profile.null_count;
            let priority = if *pct > 0.5 {
                SuggestionPriority::Critical
            } else if *pct > 0.2 {
                SuggestionPriority::High
            } else {
                SuggestionPriority::Medium
            };

            // Determine best fill strategy based on column type
            let (strategy, fill_value, reasoning) = determine_fill_strategy(col_profile);

            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Handle missing values in '{}'", column),
                description: format!(
                    "Column '{}' has {:.1}% null values ({} rows). {}",
                    column,
                    pct * 100.0,
                    null_count,
                    if *pct > 0.5 {
                        "Consider dropping this column or investigating why so many values are missing."
                    } else {
                        "Fill missing values or drop affected rows."
                    }
                ),
                addresses_issue: issue.description(),
                priority,
                category: CleaningCategory::MissingValues,
                columns: Some(vec![column.clone()]),
                operation: if *pct > 0.7 {
                    "drop_column".to_string()
                } else {
                    "fill_na".to_string()
                },
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: fill_value,
                    additional_columns: None,
                    strategy: Some(strategy),
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: null_count,
                    rows_affected_pct: *pct * 100.0,
                    completeness_change: Some(*pct / profile.columns.len() as f64 * 100.0),
                    impact_description: format!(
                        "Will affect {} rows ({:.1}%)",
                        null_count,
                        pct * 100.0
                    ),
                },
                reasoning,
                considerations: vec![
                    "Missing values may be informative (not missing at random)".to_string(),
                    "Consider the impact on downstream analysis".to_string(),
                    if *pct > 0.3 {
                        "High null rate may indicate a data collection issue".to_string()
                    } else {
                        String::new()
                    },
                ]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect(),
            })
        }

        DataIssue::WhitespaceIssues { column, count } => {
            let pct = *count as f64 / profile.row_count as f64 * 100.0;
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Trim whitespace in '{}'", column),
                description: format!(
                    "Remove leading/trailing whitespace from {} values ({:.1}%) in column '{}'.",
                    count, pct, column
                ),
                addresses_issue: issue.description(),
                priority: SuggestionPriority::High, // Whitespace issues often cause join/match failures
                category: CleaningCategory::StringFormatting,
                columns: Some(vec![column.clone()]),
                operation: "trim".to_string(),
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: None,
                    additional_columns: None,
                    strategy: None,
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: *count,
                    rows_affected_pct: pct,
                    completeness_change: None,
                    impact_description: format!("Will clean {} values", count),
                },
                reasoning: "Whitespace issues commonly cause string matching and join failures. Trimming is safe and improves data consistency.".to_string(),
                considerations: vec![
                    "Trimming is a safe, non-destructive operation".to_string(),
                    "This fix is highly recommended for any string column".to_string(),
                ],
            })
        }

        DataIssue::EmptyStrings { column, count } => {
            let pct = *count as f64 / profile.row_count as f64 * 100.0;
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Handle empty strings in '{}'", column),
                description: format!(
                    "Convert {} empty strings ({:.1}%) in column '{}' to null values for consistent missing value handling.",
                    count, pct, column
                ),
                addresses_issue: issue.description(),
                priority: SuggestionPriority::Medium,
                category: CleaningCategory::StringFormatting,
                columns: Some(vec![column.clone()]),
                operation: "replace".to_string(),
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: Some("".to_string()),
                    additional_columns: None,
                    strategy: Some("null".to_string()), // Replace empty with null
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: *count,
                    rows_affected_pct: pct,
                    completeness_change: Some(-pct), // Will decrease completeness
                    impact_description: format!("Will convert {} empty strings to null", count),
                },
                reasoning: "Empty strings and null values should be treated consistently. Converting empties to null makes missing value handling uniform.".to_string(),
                considerations: vec![
                    "This will change completeness metrics".to_string(),
                    "Some analyses may treat empty strings differently than nulls".to_string(),
                ],
            })
        }

        DataIssue::OutlierValues {
            column,
            count,
            lower_bound,
            upper_bound,
        } => {
            let pct = *count as f64 / profile.row_count as f64 * 100.0;
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Review outliers in '{}'", column),
                description: format!(
                    "Found {} outlier values ({:.1}%) in column '{}' outside the expected range [{:.2}, {:.2}].",
                    count, pct, column, lower_bound, upper_bound
                ),
                addresses_issue: issue.description(),
                priority: if pct > 5.0 { SuggestionPriority::High } else { SuggestionPriority::Medium },
                category: CleaningCategory::OutlierHandling,
                columns: Some(vec![column.clone()]),
                operation: "filter".to_string(), // Or could be "clip" to winsorize
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: Some(format!("between {} and {}", lower_bound, upper_bound)),
                    additional_columns: None,
                    strategy: Some("remove".to_string()),
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: *count,
                    rows_affected_pct: pct,
                    completeness_change: None,
                    impact_description: format!("May remove up to {} rows", count),
                },
                reasoning: "Outliers can significantly impact statistical analyses. Review these values to determine if they are errors or valid extreme observations.".to_string(),
                considerations: vec![
                    "Outliers may be valid data points that shouldn't be removed".to_string(),
                    "Consider winsorizing (capping) instead of removing".to_string(),
                    "Review the actual outlier values before deciding".to_string(),
                ],
            })
        }

        DataIssue::InconsistentFormat { column, patterns } => {
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Standardize format in '{}'", column),
                description: format!(
                    "Column '{}' has multiple formats detected: {}. Consider standardizing to a single format.",
                    column,
                    patterns.join(", ")
                ),
                addresses_issue: issue.description(),
                priority: SuggestionPriority::Medium,
                category: CleaningCategory::TypeStandardization,
                columns: Some(vec![column.clone()]),
                operation: "standardize".to_string(),
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: Some(patterns.first().cloned().unwrap_or_default()),
                    additional_columns: None,
                    strategy: Some("format".to_string()),
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: profile.row_count, // Unknown exact count
                    rows_affected_pct: 100.0,
                    completeness_change: None,
                    impact_description: "Will standardize formatting across all values".to_string(),
                },
                reasoning:
                    "Mixed formats can cause issues with parsing, sorting, and grouping operations."
                        .to_string(),
                considerations: vec![
                    "Determine which format is most appropriate for your use case".to_string(),
                    "Date formats in particular need careful standardization".to_string(),
                ],
            })
        }

        DataIssue::ConstantColumn { column, value } => Some(CleaningSuggestion {
            id: next_id(),
            title: format!("Consider removing constant column '{}'", column),
            description: format!(
                "Column '{}' contains only one value ('{}') and provides no information for analysis.",
                column, value
            ),
            addresses_issue: issue.description(),
            priority: SuggestionPriority::Low,
            category: CleaningCategory::Validation,
            columns: Some(vec![column.clone()]),
            operation: "drop_column".to_string(),
            parameters: SuggestionParameters {
                column: Some(column.clone()),
                value: None,
                additional_columns: None,
                strategy: None,
            },
            estimated_impact: EstimatedImpact {
                rows_affected: 0,
                rows_affected_pct: 0.0,
                completeness_change: None,
                impact_description: "Will remove one column".to_string(),
            },
            reasoning:
                "Constant columns add no variance and can be removed to simplify the dataset."
                    .to_string(),
            considerations: vec![
                "The constant value may be important metadata to preserve separately".to_string(),
                "Ensure this column won't have variation in future data".to_string(),
            ],
        }),

        DataIssue::MixedTypes { column, examples } => {
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Resolve mixed types in '{}'", column),
                description: format!(
                    "Column '{}' appears to contain mixed data types. Examples: {}",
                    column, examples.join(", ")
                ),
                addresses_issue: issue.description(),
                priority: SuggestionPriority::High,
                category: CleaningCategory::TypeStandardization,
                columns: Some(vec![column.clone()]),
                operation: "cast".to_string(),
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: None, // User should decide target type
                    additional_columns: None,
                    strategy: Some("infer".to_string()),
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: profile.row_count,
                    rows_affected_pct: 100.0,
                    completeness_change: None,
                    impact_description: "Will attempt to standardize all values to a single type".to_string(),
                },
                reasoning: "Mixed types cause issues with calculations and comparisons. Standardize to a single type.".to_string(),
                considerations: vec![
                    "Determine the intended type for this column".to_string(),
                    "Values that can't convert will become null".to_string(),
                ],
            })
        }

        DataIssue::HighCardinality { column, unique_pct } => {
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Note: '{}' may be an ID column", column),
                description: format!(
                    "Column '{}' has {:.1}% unique values, suggesting it may be an identifier column.",
                    column, unique_pct * 100.0
                ),
                addresses_issue: issue.description(),
                priority: SuggestionPriority::Low,
                category: CleaningCategory::Validation,
                columns: Some(vec![column.clone()]),
                operation: "none".to_string(), // Informational only
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: None,
                    additional_columns: None,
                    strategy: None,
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: 0,
                    rows_affected_pct: 0.0,
                    completeness_change: None,
                    impact_description: "No action needed - informational".to_string(),
                },
                reasoning: "High cardinality columns are typically identifiers. Exclude from statistical analyses but keep for record linking.".to_string(),
                considerations: vec![
                    "Verify this is indeed an ID column".to_string(),
                    "Ensure IDs are unique if they should be".to_string(),
                ],
            })
        }

        DataIssue::LowCardinality {
            column,
            unique_count,
            unique_pct,
        } => {
            Some(CleaningSuggestion {
                id: next_id(),
                title: format!("Consider encoding '{}' as categorical", column),
                description: format!(
                    "Column '{}' has only {} unique values ({:.1}%), making it suitable for categorical encoding.",
                    column, unique_count, unique_pct * 100.0
                ),
                addresses_issue: issue.description(),
                priority: SuggestionPriority::Low,
                category: CleaningCategory::TypeStandardization,
                columns: Some(vec![column.clone()]),
                operation: "none".to_string(), // Informational
                parameters: SuggestionParameters {
                    column: Some(column.clone()),
                    value: None,
                    additional_columns: None,
                    strategy: None,
                },
                estimated_impact: EstimatedImpact {
                    rows_affected: 0,
                    rows_affected_pct: 0.0,
                    completeness_change: None,
                    impact_description: "No action needed - informational".to_string(),
                },
                reasoning: "Low cardinality columns may be categorical variables. Consider encoding for memory efficiency.".to_string(),
                considerations: vec![
                    "Categorical encoding can improve performance".to_string(),
                ],
            })
        }

        // Dataset-level issues are handled by suggestion_for_dataset_issue, not here
        DataIssue::DuplicateRows { .. } | DataIssue::PossibleDuplicates { .. } => None,
    }
}

/// Determine the best fill strategy based on column profile.
fn determine_fill_strategy(col_profile: &ColumnProfile) -> (String, Option<String>, String) {
    if let Some(ref numeric_stats) = col_profile.numeric_stats {
        // For numeric columns, suggest median (more robust to outliers)
        (
            "median".to_string(),
            Some(format!("{:.4}", numeric_stats.median)),
            format!(
                "For numeric data, filling with the median ({:.2}) is recommended as it's robust to outliers. The mean ({:.2}) is an alternative if data is normally distributed.",
                numeric_stats.median, numeric_stats.mean
            ),
        )
    } else if let Some(ref string_stats) = col_profile.string_stats {
        // For string columns, suggest mode (most common value) if available
        if let Some((most_common, _)) = string_stats.top_values.first() {
            (
                "mode".to_string(),
                Some(most_common.clone()),
                format!(
                    "For categorical/string data, filling with the most common value ('{}') preserves the distribution.",
                    most_common
                ),
            )
        } else {
            (
                "constant".to_string(),
                Some("Unknown".to_string()),
                "For string data with no clear mode, consider filling with a placeholder value."
                    .to_string(),
            )
        }
    } else {
        (
            "drop".to_string(),
            None,
            "For this column type, dropping rows with missing values may be the safest option."
                .to_string(),
        )
    }
}

/// Generate an overall recommendation based on suggestions.
fn generate_overall_recommendation(
    suggestions: &[CleaningSuggestion],
    profile: &DataQualityProfile,
) -> String {
    if suggestions.is_empty() {
        return "Your dataset appears to be clean! No immediate cleaning actions are recommended."
            .to_string();
    }

    let critical_count = suggestions
        .iter()
        .filter(|s| s.priority == SuggestionPriority::Critical)
        .count();
    let high_count = suggestions
        .iter()
        .filter(|s| s.priority == SuggestionPriority::High)
        .count();

    let mut rec = String::new();

    if critical_count > 0 {
        rec.push_str(&format!(
            "⚠️ Found {} critical issue(s) that should be addressed before analysis.\n",
            critical_count
        ));
    }

    if high_count > 0 {
        rec.push_str(&format!(
            "Found {} high-priority issue(s) that are recommended to fix.\n",
            high_count
        ));
    }

    // Suggest starting point
    if let Some(first) = suggestions.first() {
        rec.push_str(&format!("\nRecommended starting point: {}\n", first.title));
    }

    // Add completeness note
    if profile.completeness_score < 0.9 {
        rec.push_str(&format!(
            "\nDataset completeness is {:.1}%. Consider addressing missing values to improve data quality.\n",
            profile.completeness_score * 100.0
        ));
    }

    rec
}

impl SuggestionReport {
    /// Get a human-readable summary of suggestions.
    pub fn summary(&self) -> String {
        let mut summary = format!(
            "Cleaning Suggestions Report\n\
             ============================\n\
             Dataset: {} rows × {} columns\n\
             Completeness: {:.1}%\n\
             Issues analyzed: {}\n\
             Suggestions generated: {}\n\n",
            self.dataset_summary.row_count,
            self.dataset_summary.column_count,
            self.dataset_summary.completeness_score * 100.0,
            self.issues_analyzed,
            self.suggestions.len(),
        );

        if self.suggestions.is_empty() {
            summary.push_str("✓ No cleaning suggestions - your data looks clean!\n");
        } else {
            // Group by priority
            for priority in [
                SuggestionPriority::Critical,
                SuggestionPriority::High,
                SuggestionPriority::Medium,
                SuggestionPriority::Low,
            ] {
                let priority_suggestions: Vec<_> = self
                    .suggestions
                    .iter()
                    .filter(|s| s.priority == priority)
                    .collect();

                if !priority_suggestions.is_empty() {
                    summary.push_str(&format!(
                        "\n{} Priority ({}):\n",
                        priority.label(),
                        priority_suggestions.len()
                    ));
                    for (i, s) in priority_suggestions.iter().enumerate() {
                        summary.push_str(&format!(
                            "  {}. {} ({})\n",
                            i + 1,
                            s.title,
                            s.estimated_impact.impact_description
                        ));
                    }
                }
            }

            summary.push_str(&format!("\n{}\n", self.overall_recommendation));
        }

        summary
    }

    /// Get suggestions filtered by priority.
    pub fn by_priority(&self, priority: SuggestionPriority) -> Vec<&CleaningSuggestion> {
        self.suggestions
            .iter()
            .filter(|s| s.priority == priority)
            .collect()
    }

    /// Get suggestions filtered by category.
    pub fn by_category(&self, category: &CleaningCategory) -> Vec<&CleaningSuggestion> {
        self.suggestions
            .iter()
            .filter(|s| &s.category == category)
            .collect()
    }

    /// Get suggestions for a specific column.
    pub fn for_column(&self, column: &str) -> Vec<&CleaningSuggestion> {
        self.suggestions
            .iter()
            .filter(|s| {
                s.columns
                    .as_ref()
                    .is_some_and(|cols| cols.contains(&column.to_string()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, generate_quality_profile};
    use polars::prelude::df;

    #[test]
    fn test_generate_suggestions_clean_data() {
        let test_df = df! {
            "id" => [1, 2, 3, 4, 5],
            "name" => ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score" => [85.5, 90.0, 78.5, 92.0, 88.0],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        // Clean data should have few or no suggestions
        assert!(report.suggestions.len() <= 2); // May have low cardinality notes
    }

    #[test]
    fn test_suggestion_for_whitespace() {
        let test_df = df! {
            "email" => ["  alice@test.com", "bob@test.com  ", " charlie@test.com ", "diana@test.com"],
        }.unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        // Should suggest trimming whitespace
        let trim_suggestion = report.suggestions.iter().find(|s| s.operation == "trim");
        assert!(trim_suggestion.is_some());

        let s = trim_suggestion.unwrap();
        assert_eq!(s.priority, SuggestionPriority::High);
        assert_eq!(s.category, CleaningCategory::StringFormatting);
    }

    #[test]
    fn test_suggestion_for_nulls() {
        let test_df = df! {
            "id" => [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "value" => [Some(1.0), Some(2.0), None, None, None, None, Some(7.0), None, None, None],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        // Should suggest handling missing values
        let null_suggestion = report
            .suggestions
            .iter()
            .find(|s| s.category == CleaningCategory::MissingValues);
        assert!(null_suggestion.is_some());
    }

    #[test]
    fn test_suggestion_for_duplicates() {
        let test_df = df! {
            "id" => [1, 2, 2, 3, 3, 3],
            "name" => ["A", "B", "B", "C", "C", "C"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        // Should suggest deduplication
        let dedup_suggestion = report
            .suggestions
            .iter()
            .find(|s| s.operation == "deduplicate");
        assert!(dedup_suggestion.is_some());

        let s = dedup_suggestion.unwrap();
        assert_eq!(s.category, CleaningCategory::Deduplication);
    }

    #[test]
    fn test_suggestion_for_outliers() {
        // Create data with clear outliers
        let test_df = df! {
            "value" => [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 1000.0],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        // Should suggest reviewing outliers
        let outlier_suggestion = report
            .suggestions
            .iter()
            .find(|s| s.category == CleaningCategory::OutlierHandling);
        assert!(outlier_suggestion.is_some());
    }

    #[test]
    fn test_suggestion_priority_ordering() {
        // Create data with multiple issues
        let test_df = df! {
            "email" => ["  alice@test.com", "bob@test.com  ", " charlie@test.com "],
            "value" => [Some(1.0), None, None], // 66% null - should be high/critical priority
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        // Suggestions should be ordered by priority
        if report.suggestions.len() >= 2 {
            assert!(report.suggestions[0].priority >= report.suggestions[1].priority);
        }
    }

    #[test]
    fn test_suggestion_report_summary() {
        let test_df = df! {
            "email" => ["  test@example.com", "test2@example.com"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        let summary = report.summary();
        assert!(summary.contains("Cleaning Suggestions Report"));
        assert!(summary.contains("rows"));
        assert!(summary.contains("columns"));
    }

    #[test]
    fn test_filter_by_priority() {
        let test_df = df! {
            "email" => ["  alice@test.com", "bob@test.com"],
            "value" => [Some(1.0), None],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        let high_priority = report.by_priority(SuggestionPriority::High);
        for s in high_priority {
            assert_eq!(s.priority, SuggestionPriority::High);
        }
    }

    #[test]
    fn test_filter_by_column() {
        let test_df = df! {
            "email" => ["  alice@test.com", "bob@test.com"],
            "name" => ["Alice", "Bob"],
        }
        .unwrap();

        let dataset = Dataset::new(test_df);
        let profile = generate_quality_profile(&dataset);
        let report = generate_suggestions(&profile);

        let email_suggestions = report.for_column("email");
        for s in email_suggestions {
            assert!(s.columns.as_ref().unwrap().contains(&"email".to_string()));
        }
    }
}
