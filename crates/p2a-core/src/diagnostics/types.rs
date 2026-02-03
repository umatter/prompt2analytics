//! Core types for identification diagnostics.

use serde::{Deserialize, Serialize};

/// Severity level for identification warnings.
///
/// Warnings are non-blocking regardless of severity; they inform interpretation
/// but do not prevent analysis execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WarningSeverity {
    /// Informational: assumption stated, no evidence of violation
    Info,
    /// Caution: some indicators suggest potential issues worth noting
    Caution,
    /// Warning: evidence suggests assumption may be violated
    Warning,
    /// Critical: strong evidence of assumption violation; interpret with care
    Critical,
}

impl std::fmt::Display for WarningSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WarningSeverity::Info => write!(f, "Info"),
            WarningSeverity::Caution => write!(f, "Caution"),
            WarningSeverity::Warning => write!(f, "Warning"),
            WarningSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// A single identification warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationWarning {
    /// Short identifier (e.g., "WEAK_INSTRUMENT", "PARALLEL_TRENDS_VIOLATED")
    pub code: String,
    /// Severity level
    pub severity: WarningSeverity,
    /// Human-readable title
    pub title: String,
    /// Detailed explanation of the issue
    pub message: String,
    /// What assumption is potentially violated
    pub assumption: String,
    /// Suggested remediation steps
    pub remediation: Vec<String>,
    /// Numeric diagnostics if applicable
    pub diagnostics: Option<WarningDiagnostics>,
}

impl IdentificationWarning {
    /// Create a new warning with the given parameters.
    pub fn new(
        code: impl Into<String>,
        severity: WarningSeverity,
        title: impl Into<String>,
        message: impl Into<String>,
        assumption: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity,
            title: title.into(),
            message: message.into(),
            assumption: assumption.into(),
            remediation: Vec::new(),
            diagnostics: None,
        }
    }

    /// Add remediation suggestions.
    pub fn with_remediation(mut self, remediation: Vec<String>) -> Self {
        self.remediation = remediation;
        self
    }

    /// Add numeric diagnostics.
    pub fn with_diagnostics(mut self, diagnostics: WarningDiagnostics) -> Self {
        self.diagnostics = Some(diagnostics);
        self
    }

    /// Format warning for display.
    pub fn format_display(&self) -> String {
        let icon = match self.severity {
            WarningSeverity::Info => "ℹ️",
            WarningSeverity::Caution => "⚡",
            WarningSeverity::Warning => "⚠️",
            WarningSeverity::Critical => "🚨",
        };

        let mut output = format!(
            "{} **{}**: {}\n\n{}\n\n**Assumption:** {}",
            icon, self.severity, self.title, self.message, self.assumption
        );

        if let Some(ref diag) = self.diagnostics {
            output.push_str(&format!(
                "\n\n**Diagnostic:** {} = {:.4}",
                diag.name, diag.value
            ));
            if let Some(threshold) = diag.threshold {
                output.push_str(&format!(" (threshold: {:.4})", threshold));
            }
        }

        if !self.remediation.is_empty() {
            output.push_str("\n\n**Recommendations:**\n");
            for rec in &self.remediation {
                output.push_str(&format!("- {}\n", rec));
            }
        }

        output
    }
}

/// Numeric diagnostics associated with a warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarningDiagnostics {
    /// Name of the diagnostic (e.g., "First-stage F")
    pub name: String,
    /// Observed value
    pub value: f64,
    /// Threshold for concern (if applicable)
    pub threshold: Option<f64>,
    /// Whether value exceeds threshold in problematic direction
    pub exceeds_threshold: bool,
}

impl WarningDiagnostics {
    /// Create new diagnostics.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            threshold: None,
            exceeds_threshold: false,
        }
    }

    /// Add threshold information.
    pub fn with_threshold(mut self, threshold: f64, exceeds: bool) -> Self {
        self.threshold = Some(threshold);
        self.exceeds_threshold = exceeds;
        self
    }
}

/// Status of an identification assumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssumptionStatus {
    /// No evidence of violation based on available diagnostics
    NoViolation,
    /// Assumption cannot be tested from data
    Untestable,
    /// Some indicators suggest potential violation
    PotentialViolation,
    /// Strong evidence of violation
    LikelyViolation,
}

impl std::fmt::Display for AssumptionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssumptionStatus::NoViolation => write!(f, "No violation detected"),
            AssumptionStatus::Untestable => write!(f, "Cannot be tested"),
            AssumptionStatus::PotentialViolation => write!(f, "Potential violation"),
            AssumptionStatus::LikelyViolation => write!(f, "Likely violation"),
        }
    }
}

/// An assumption required by a causal method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assumption {
    /// Assumption name
    pub name: String,
    /// Whether this assumption is testable from data
    pub testable: bool,
    /// Brief description
    pub description: String,
    /// Status based on diagnostics
    pub status: AssumptionStatus,
}

impl Assumption {
    /// Create a new testable assumption.
    pub fn testable(
        name: impl Into<String>,
        description: impl Into<String>,
        status: AssumptionStatus,
    ) -> Self {
        Self {
            name: name.into(),
            testable: true,
            description: description.into(),
            status,
        }
    }

    /// Create a new untestable assumption.
    pub fn untestable(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            testable: false,
            description: description.into(),
            status: AssumptionStatus::Untestable,
        }
    }
}

/// Complete identification report for a causal method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationReport {
    /// Method that was run
    pub method: String,
    /// List of warnings generated
    pub warnings: Vec<IdentificationWarning>,
    /// Assumptions and their status
    pub assumptions: Vec<Assumption>,
}

impl IdentificationReport {
    /// Create a new empty report for the given method.
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            warnings: Vec::new(),
            assumptions: Vec::new(),
        }
    }

    /// Add a warning to the report.
    pub fn add_warning(&mut self, warning: IdentificationWarning) {
        self.warnings.push(warning);
    }

    /// Add an assumption to the report.
    pub fn add_assumption(&mut self, assumption: Assumption) {
        self.assumptions.push(assumption);
    }

    /// Check if any critical warnings exist.
    pub fn has_critical(&self) -> bool {
        self.warnings
            .iter()
            .any(|w| w.severity == WarningSeverity::Critical)
    }

    /// Check if any warnings at or above the given severity exist.
    pub fn has_warnings_at_level(&self, min_severity: WarningSeverity) -> bool {
        self.warnings.iter().any(|w| w.severity >= min_severity)
    }

    /// Get the maximum severity among all warnings.
    pub fn max_severity(&self) -> Option<WarningSeverity> {
        self.warnings.iter().map(|w| w.severity).max()
    }

    /// Get warnings filtered by minimum severity.
    pub fn warnings_at_level(&self, min_severity: WarningSeverity) -> Vec<&IdentificationWarning> {
        self.warnings
            .iter()
            .filter(|w| w.severity >= min_severity)
            .collect()
    }

    /// Generate a summary suitable for LLM communication.
    pub fn summary(&self) -> String {
        if self.warnings.is_empty() {
            return format!(
                "No identification concerns detected for {} analysis.",
                self.method
            );
        }

        let critical_count = self
            .warnings
            .iter()
            .filter(|w| w.severity == WarningSeverity::Critical)
            .count();
        let warning_count = self
            .warnings
            .iter()
            .filter(|w| w.severity == WarningSeverity::Warning)
            .count();
        let caution_count = self
            .warnings
            .iter()
            .filter(|w| w.severity == WarningSeverity::Caution)
            .count();

        let mut parts = Vec::new();
        if critical_count > 0 {
            parts.push(format!("{} critical", critical_count));
        }
        if warning_count > 0 {
            parts.push(format!("{} warning(s)", warning_count));
        }
        if caution_count > 0 {
            parts.push(format!("{} caution(s)", caution_count));
        }

        format!(
            "Identification diagnostics for {}: {}. Review warnings before interpreting results causally.",
            self.method,
            parts.join(", ")
        )
    }

    /// Format the full report for display.
    pub fn format_display(&self) -> String {
        let mut output = format!("# Identification Report: {}\n\n", self.method);

        // Summary
        output.push_str(&format!("{}\n\n", self.summary()));

        // Warnings
        if !self.warnings.is_empty() {
            output.push_str("## Warnings\n\n");
            for warning in &self.warnings {
                output.push_str(&warning.format_display());
                output.push_str("\n---\n\n");
            }
        }

        // Assumptions
        if !self.assumptions.is_empty() {
            output.push_str("## Assumptions\n\n");
            output.push_str("| Assumption | Testable | Status |\n");
            output.push_str("|------------|----------|--------|\n");
            for assumption in &self.assumptions {
                output.push_str(&format!(
                    "| {} | {} | {} |\n",
                    assumption.name,
                    if assumption.testable { "Yes" } else { "No" },
                    assumption.status
                ));
            }
        }

        output
    }
}

impl Default for IdentificationReport {
    fn default() -> Self {
        Self::new("Unknown")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warning_severity_ordering() {
        assert!(WarningSeverity::Info < WarningSeverity::Caution);
        assert!(WarningSeverity::Caution < WarningSeverity::Warning);
        assert!(WarningSeverity::Warning < WarningSeverity::Critical);
    }

    #[test]
    fn test_identification_report_summary() {
        let mut report = IdentificationReport::new("2SLS");

        // Empty report
        assert!(report.summary().contains("No identification concerns"));

        // Add a warning
        report.add_warning(IdentificationWarning::new(
            "WEAK_INSTRUMENT",
            WarningSeverity::Warning,
            "Weak Instrument",
            "F-stat below 10",
            "Instrument relevance",
        ));

        assert!(report.summary().contains("1 warning"));
        assert!(report.has_warnings_at_level(WarningSeverity::Warning));
        assert!(!report.has_critical());
    }

    #[test]
    fn test_warning_format() {
        let warning = IdentificationWarning::new(
            "TEST",
            WarningSeverity::Warning,
            "Test Warning",
            "This is a test",
            "Test assumption",
        )
        .with_diagnostics(WarningDiagnostics::new("Test stat", 5.0).with_threshold(10.0, false))
        .with_remediation(vec!["Fix it".to_string()]);

        let formatted = warning.format_display();
        assert!(formatted.contains("⚠️"));
        assert!(formatted.contains("Test Warning"));
        assert!(formatted.contains("Test stat"));
        assert!(formatted.contains("Fix it"));
    }
}
