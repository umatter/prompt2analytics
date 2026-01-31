# Design Document: Identification Warnings System

**Author:** prompt2analytics team
**Date:** January 30, 2026
**Status:** Draft

---

## 1. Overview

### 1.1 Motivation

Users of chat-first analytics may request causal effect estimates without fully understanding the identification assumptions required. The system currently executes requested methods and returns results without validating whether causal interpretations are warranted. This creates risk of:

- Users interpreting correlations as causal effects
- Overlooking violations of key assumptions (SUTVA, parallel trends, exclusion restriction)
- Publishing or acting on misleading analyses

### 1.2 Design Goals

1. **Proactive warnings**: Alert users to potential identification problems before they misinterpret results
2. **Educational**: Help users understand what assumptions their analysis requires
3. **Non-blocking**: Warnings inform but don't prevent analysis execution
4. **Method-appropriate**: Each method gets relevant diagnostics, not generic warnings
5. **Configurable**: Power users can suppress warnings; novices see full guidance

### 1.3 Non-Goals

- Replacing statistical judgment (users remain responsible for research design)
- Guaranteeing correct identification (impossible without domain knowledge)
- Blocking analyses that might be valid in context

---

## 2. Warning Categories

### 2.1 Testable Assumptions (Data-Driven)

These can be evaluated algorithmically from the data:

| Assumption | Methods | Diagnostic |
|------------|---------|------------|
| Positivity/Overlap | IPW, Matching, DiD | Propensity score distribution |
| No manipulation | RD | McCrary density test |
| Parallel pre-trends | DiD | Pre-treatment coefficient test |
| Instrument strength | IV/2SLS | First-stage F-statistic |
| Covariate balance | Matching, IPW | Standardized mean differences |
| Common support | Matching | Off-support unit count |

### 2.2 Untestable Assumptions (Heuristic Warnings)

These cannot be tested but risk factors can be flagged:

| Assumption | Methods | Risk Indicators |
|------------|---------|-----------------|
| SUTVA (no spillovers) | All causal | Geographic clustering, network structure |
| SUTVA (no hidden variation) | All causal | Treatment intensity variation |
| Exclusion restriction | IV | (Requires domain knowledge) |
| Unconfoundedness | Matching, IPW | High-dimensional covariate space |
| No anticipation | DiD, Event study | Pre-trend patterns |

### 2.3 Interpretation Warnings (LLM-Layer)

These address how results are communicated:

| Concern | Trigger | Response |
|---------|---------|----------|
| Causal language without ID strategy | User says "effect of X on Y" | Ask about identification |
| Coefficient as causal | OLS without controls discussion | Add associational caveat |
| Significance fishing | Multiple specifications requested | Warn about multiple testing |

---

## 3. Technical Design

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCP Server                               │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Tool Call   │───▶│ Estimation   │───▶│ Warning Engine    │  │
│  │ Handler     │    │ (p2a-core)   │    │ (post-estimation) │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│         │                                        │              │
│         │                                        ▼              │
│         │           ┌──────────────────────────────────────┐   │
│         │           │ IdentificationReport                  │   │
│         │           │ - warnings: Vec<IdentificationWarning>│   │
│         │           │ - diagnostics: MethodDiagnostics      │   │
│         │           │ - assumptions: Vec<Assumption>        │   │
│         │           └──────────────────────────────────────┘   │
│         │                                        │              │
│         ▼                                        ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    LLM Response                          │   │
│  │  - Results (coefficients, SEs, etc.)                     │   │
│  │  - Warnings (formatted for user)                         │   │
│  │  - Diagnostic plots (if applicable)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Data Structures

```rust
// crates/p2a-core/src/diagnostics/identification.rs

use serde::{Deserialize, Serialize};

/// Severity level for identification warnings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningSeverity {
    /// Informational: assumption stated, no evidence of violation
    Info,
    /// Caution: some indicators suggest potential issues
    Caution,
    /// Warning: strong evidence of assumption violation
    Warning,
    /// Critical: analysis likely invalid without addressing
    Critical,
}

/// A single identification warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationWarning {
    /// Short identifier (e.g., "SUTVA_SPILLOVER", "WEAK_INSTRUMENT")
    pub code: String,
    /// Severity level
    pub severity: WarningSeverity,
    /// Human-readable title
    pub title: String,
    /// Detailed explanation
    pub message: String,
    /// What assumption is potentially violated
    pub assumption: String,
    /// Suggested remediation steps
    pub remediation: Vec<String>,
    /// Numeric diagnostics if applicable (e.g., F-stat value)
    pub diagnostics: Option<WarningDiagnostics>,
}

/// Numeric diagnostics associated with a warning
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

/// Assumptions required by a method
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssumptionStatus {
    /// No evidence of violation
    NoViolation,
    /// Unable to test
    Untestable,
    /// Some concern
    PotentialViolation,
    /// Strong evidence of violation
    LikelyViolation,
}

/// Complete identification report for a causal method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentificationReport {
    /// Method that was run
    pub method: String,
    /// List of warnings generated
    pub warnings: Vec<IdentificationWarning>,
    /// Assumptions and their status
    pub assumptions: Vec<Assumption>,
    /// Whether any critical warnings exist
    pub has_critical: bool,
    /// Summary suitable for LLM to communicate
    pub summary: String,
}
```

### 3.3 Warning Generators

Each method family gets a warning generator:

```rust
// crates/p2a-core/src/diagnostics/generators/mod.rs

pub mod iv_warnings;
pub mod did_warnings;
pub mod matching_warnings;
pub mod rd_warnings;
pub mod panel_warnings;

/// Trait for methods that can generate identification warnings
pub trait IdentificationCheck {
    /// Generate identification report after estimation
    fn check_identification(&self, data: &Dataset) -> IdentificationReport;
}
```

### 3.4 Method-Specific Implementations

#### 3.4.1 Instrumental Variables (IV/2SLS)

```rust
// crates/p2a-core/src/diagnostics/generators/iv_warnings.rs

impl IdentificationCheck for TwoSlsResult {
    fn check_identification(&self, data: &Dataset) -> IdentificationReport {
        let mut warnings = Vec::new();
        let mut assumptions = Vec::new();

        // 1. Weak instrument check
        let f_stat = self.first_stage_f();
        if f_stat < 10.0 {
            warnings.push(IdentificationWarning {
                code: "WEAK_INSTRUMENT".to_string(),
                severity: if f_stat < 5.0 {
                    WarningSeverity::Critical
                } else {
                    WarningSeverity::Warning
                },
                title: "Weak Instrument Detected".to_string(),
                message: format!(
                    "First-stage F-statistic is {:.2}, below the rule-of-thumb \
                     threshold of 10. Weak instruments cause bias toward OLS \
                     estimates and unreliable inference.",
                    f_stat
                ),
                assumption: "Instrument relevance".to_string(),
                remediation: vec![
                    "Consider stronger instruments".to_string(),
                    "Use weak-instrument-robust inference (Anderson-Rubin)".to_string(),
                    "Report reduced-form estimates as robustness check".to_string(),
                ],
                diagnostics: Some(WarningDiagnostics {
                    name: "First-stage F".to_string(),
                    value: f_stat,
                    threshold: Some(10.0),
                    exceeds_threshold: false,
                }),
            });
        }

        assumptions.push(Assumption {
            name: "Instrument relevance".to_string(),
            testable: true,
            description: "Instruments must predict the endogenous variable".to_string(),
            status: if f_stat >= 10.0 {
                AssumptionStatus::NoViolation
            } else {
                AssumptionStatus::LikelyViolation
            },
        });

        // 2. Exclusion restriction (untestable, but state it)
        assumptions.push(Assumption {
            name: "Exclusion restriction".to_string(),
            testable: false,
            description: "Instruments affect outcome only through the endogenous variable".to_string(),
            status: AssumptionStatus::Untestable,
        });

        // 3. Overidentification test (if overidentified)
        if self.n_instruments() > self.n_endogenous() {
            let sargan = self.sargan_test();
            if sargan.p_value < 0.05 {
                warnings.push(IdentificationWarning {
                    code: "OVERID_REJECTED".to_string(),
                    severity: WarningSeverity::Warning,
                    title: "Overidentification Test Rejected".to_string(),
                    message: format!(
                        "Sargan test rejects the null (p = {:.4}). At least one \
                         instrument may violate the exclusion restriction.",
                        sargan.p_value
                    ),
                    assumption: "Exclusion restriction".to_string(),
                    remediation: vec![
                        "Examine each instrument's validity".to_string(),
                        "Consider dropping suspect instruments".to_string(),
                    ],
                    diagnostics: Some(WarningDiagnostics {
                        name: "Sargan p-value".to_string(),
                        value: sargan.p_value,
                        threshold: Some(0.05),
                        exceeds_threshold: true,
                    }),
                });
            }
        }

        // Build report
        let has_critical = warnings.iter().any(|w| w.severity == WarningSeverity::Critical);

        IdentificationReport {
            method: "2SLS".to_string(),
            warnings,
            assumptions,
            has_critical,
            summary: self.generate_summary(),
        }
    }
}
```

#### 3.4.2 Difference-in-Differences

```rust
// crates/p2a-core/src/diagnostics/generators/did_warnings.rs

impl IdentificationCheck for DidResult {
    fn check_identification(&self, data: &Dataset) -> IdentificationReport {
        let mut warnings = Vec::new();
        let mut assumptions = Vec::new();

        // 1. Parallel trends test (if pre-periods available)
        if let Some(pretrend_test) = self.parallel_trends_test() {
            assumptions.push(Assumption {
                name: "Parallel trends".to_string(),
                testable: true,  // testable in pre-period
                description: "Treatment and control would follow same trend absent treatment".to_string(),
                status: if pretrend_test.p_value > 0.05 {
                    AssumptionStatus::NoViolation
                } else {
                    AssumptionStatus::PotentialViolation
                },
            });

            if pretrend_test.p_value < 0.05 {
                warnings.push(IdentificationWarning {
                    code: "PARALLEL_TRENDS_VIOLATED".to_string(),
                    severity: WarningSeverity::Warning,
                    title: "Pre-Trends Differ Significantly".to_string(),
                    message: format!(
                        "Joint test of pre-treatment coefficients rejects parallel \
                         trends (p = {:.4}). Treatment and control groups may have \
                         been on different trajectories before treatment.",
                        pretrend_test.p_value
                    ),
                    assumption: "Parallel trends".to_string(),
                    remediation: vec![
                        "Examine event study plot for divergence patterns".to_string(),
                        "Consider matching on pre-trends".to_string(),
                        "Use synthetic control as alternative".to_string(),
                        "Report parallel trends plot in appendix".to_string(),
                    ],
                    diagnostics: Some(WarningDiagnostics {
                        name: "Pre-trend F-test p-value".to_string(),
                        value: pretrend_test.p_value,
                        threshold: Some(0.05),
                        exceeds_threshold: true,
                    }),
                });
            }
        }

        // 2. No anticipation (check for pre-treatment effects)
        if let Some(anticipation) = self.detect_anticipation() {
            if anticipation.detected {
                warnings.push(IdentificationWarning {
                    code: "ANTICIPATION_DETECTED".to_string(),
                    severity: WarningSeverity::Caution,
                    title: "Possible Anticipation Effects".to_string(),
                    message: format!(
                        "Significant effects detected {} periods before treatment. \
                         Units may have anticipated treatment and changed behavior.",
                        anticipation.periods_before
                    ),
                    assumption: "No anticipation".to_string(),
                    remediation: vec![
                        "Verify treatment timing is correctly coded".to_string(),
                        "Consider earlier effective treatment date".to_string(),
                        "Allow for anticipation in Callaway-Sant'Anna estimator".to_string(),
                    ],
                    diagnostics: None,
                });
            }
        }

        // 3. SUTVA checks
        self.check_sutva(data, &mut warnings, &mut assumptions);

        // 4. Composition changes
        if let Some(composition) = self.check_composition_changes(data) {
            if composition.significant_change {
                warnings.push(IdentificationWarning {
                    code: "COMPOSITION_CHANGE".to_string(),
                    severity: WarningSeverity::Caution,
                    title: "Sample Composition Changes Over Time".to_string(),
                    message: "The composition of treated or control groups changes \
                              significantly over time, which may confound the treatment effect.".to_string(),
                    assumption: "Stable composition".to_string(),
                    remediation: vec![
                        "Use balanced panel if possible".to_string(),
                        "Check for differential attrition".to_string(),
                    ],
                    diagnostics: None,
                });
            }
        }

        IdentificationReport {
            method: "Difference-in-Differences".to_string(),
            warnings,
            assumptions,
            has_critical: warnings.iter().any(|w| w.severity == WarningSeverity::Critical),
            summary: self.generate_summary(),
        }
    }
}
```

#### 3.4.3 SUTVA Checks (Cross-Method)

```rust
// crates/p2a-core/src/diagnostics/sutva.rs

use crate::data::Dataset;

/// Configuration for SUTVA checks
pub struct SutvaCheckConfig {
    /// Column containing geographic coordinates (lat, lon) or location ID
    pub location_col: Option<String>,
    /// Column containing group/cluster membership
    pub group_col: Option<String>,
    /// Column containing network edges or connections
    pub network_col: Option<String>,
    /// Treatment column
    pub treatment_col: String,
    /// Distance threshold for "nearby" units (if geographic)
    pub proximity_threshold_km: f64,
}

/// Results of SUTVA diagnostic checks
pub struct SutvaCheckResult {
    /// Potential spillover risk indicators
    pub spillover_risk: SpilloverRisk,
    /// Treatment variation concerns
    pub treatment_variation: TreatmentVariationRisk,
    /// Generated warnings
    pub warnings: Vec<IdentificationWarning>,
}

#[derive(Debug, Clone)]
pub struct SpilloverRisk {
    /// Fraction of control units "near" treated units
    pub control_near_treated: f64,
    /// Average number of treated neighbors per control unit
    pub avg_treated_neighbors: f64,
    /// Whether geographic clustering is detected
    pub geographic_clustering: bool,
    /// Whether network clustering is detected
    pub network_clustering: bool,
}

pub fn check_sutva(
    data: &Dataset,
    config: &SutvaCheckConfig,
) -> SutvaCheckResult {
    let mut warnings = Vec::new();
    let mut spillover_risk = SpilloverRisk::default();

    // 1. Geographic proximity check
    if let Some(ref loc_col) = config.location_col {
        if let Ok(proximity) = check_geographic_proximity(data, loc_col, &config.treatment_col, config.proximity_threshold_km) {
            spillover_risk.control_near_treated = proximity.fraction_near;
            spillover_risk.geographic_clustering = proximity.is_clustered;

            if proximity.fraction_near > 0.3 {
                warnings.push(IdentificationWarning {
                    code: "SUTVA_GEOGRAPHIC_SPILLOVER".to_string(),
                    severity: if proximity.fraction_near > 0.5 {
                        WarningSeverity::Warning
                    } else {
                        WarningSeverity::Caution
                    },
                    title: "Potential Geographic Spillovers".to_string(),
                    message: format!(
                        "{:.0}% of control units are within {:.0} km of treated units. \
                         Treatment effects may spill over to nearby controls, biasing \
                         estimates toward zero.",
                        proximity.fraction_near * 100.0,
                        config.proximity_threshold_km
                    ),
                    assumption: "SUTVA: No spillovers".to_string(),
                    remediation: vec![
                        "Consider spatial spillover models".to_string(),
                        "Use ring/donut designs excluding border regions".to_string(),
                        "Cluster treatment at higher geographic level".to_string(),
                        "Estimate spillover effects directly".to_string(),
                    ],
                    diagnostics: Some(WarningDiagnostics {
                        name: "Fraction of controls near treated".to_string(),
                        value: proximity.fraction_near,
                        threshold: Some(0.3),
                        exceeds_threshold: true,
                    }),
                });
            }
        }
    }

    // 2. Group/cluster membership check
    if let Some(ref group_col) = config.group_col {
        if let Ok(clustering) = check_within_group_spillover(data, group_col, &config.treatment_col) {
            if clustering.mixed_groups_fraction > 0.2 {
                warnings.push(IdentificationWarning {
                    code: "SUTVA_WITHIN_GROUP_SPILLOVER".to_string(),
                    severity: WarningSeverity::Caution,
                    title: "Potential Within-Group Spillovers".to_string(),
                    message: format!(
                        "{:.0}% of groups contain both treated and control units. \
                         Treatment may spill over within groups (e.g., firms in same \
                         industry, students in same school).",
                        clustering.mixed_groups_fraction * 100.0
                    ),
                    assumption: "SUTVA: No spillovers".to_string(),
                    remediation: vec![
                        "Cluster treatment at group level".to_string(),
                        "Include group-level treatment intensity as control".to_string(),
                        "Estimate peer effects explicitly".to_string(),
                    ],
                    diagnostics: None,
                });
            }
        }
    }

    // 3. Treatment intensity variation
    if let Ok(variation) = check_treatment_variation(data, &config.treatment_col) {
        if variation.has_intensity_variation && !variation.intensity_acknowledged {
            warnings.push(IdentificationWarning {
                code: "SUTVA_TREATMENT_VARIATION".to_string(),
                severity: WarningSeverity::Caution,
                title: "Treatment Intensity Varies".to_string(),
                message: "Treatment is not binary; intensity varies across treated units. \
                          SUTVA requires treatment to be well-defined. Consider whether \
                          different doses represent the same treatment.".to_string(),
                assumption: "SUTVA: No hidden treatment variation".to_string(),
                remediation: vec![
                    "Model treatment intensity explicitly".to_string(),
                    "Estimate dose-response relationship".to_string(),
                    "Binarize at meaningful threshold if appropriate".to_string(),
                ],
                diagnostics: None,
            });
        }
    }

    // 4. High treatment saturation
    let treatment_fraction = calculate_treatment_fraction(data, &config.treatment_col);
    if treatment_fraction > 0.5 {
        warnings.push(IdentificationWarning {
            code: "SUTVA_HIGH_SATURATION".to_string(),
            severity: WarningSeverity::Info,
            title: "High Treatment Saturation".to_string(),
            message: format!(
                "{:.0}% of units are treated. With high saturation, general equilibrium \
                 effects may occur where treatment affects market-level outcomes that \
                 impact control units.",
                treatment_fraction * 100.0
            ),
            assumption: "SUTVA: No general equilibrium effects".to_string(),
            remediation: vec![
                "Consider whether market-level effects are possible".to_string(),
                "Compare to settings with lower saturation".to_string(),
            ],
            diagnostics: None,
        });
    }

    SutvaCheckResult {
        spillover_risk,
        treatment_variation: TreatmentVariationRisk::default(),
        warnings,
    }
}
```

#### 3.4.4 Matching/IPW Warnings

```rust
// crates/p2a-core/src/diagnostics/generators/matching_warnings.rs

impl IdentificationCheck for MatchingResult {
    fn check_identification(&self, data: &Dataset) -> IdentificationReport {
        let mut warnings = Vec::new();
        let mut assumptions = Vec::new();

        // 1. Positivity/Overlap check
        let overlap = self.check_propensity_overlap();
        assumptions.push(Assumption {
            name: "Positivity (overlap)".to_string(),
            testable: true,
            description: "All covariate values must have positive probability of treatment and control".to_string(),
            status: match overlap.violation_severity {
                OverlapSeverity::None => AssumptionStatus::NoViolation,
                OverlapSeverity::Mild => AssumptionStatus::PotentialViolation,
                OverlapSeverity::Severe => AssumptionStatus::LikelyViolation,
            },
        });

        if overlap.fraction_off_support > 0.05 {
            warnings.push(IdentificationWarning {
                code: "POSITIVITY_VIOLATION".to_string(),
                severity: if overlap.fraction_off_support > 0.2 {
                    WarningSeverity::Warning
                } else {
                    WarningSeverity::Caution
                },
                title: "Limited Common Support".to_string(),
                message: format!(
                    "{:.1}% of units are off common support (propensity scores \
                     near 0 or 1). Treatment effect estimates may rely heavily on \
                     model extrapolation for these units.",
                    overlap.fraction_off_support * 100.0
                ),
                assumption: "Positivity".to_string(),
                remediation: vec![
                    "Trim sample to common support region".to_string(),
                    "Use matching instead of weighting".to_string(),
                    "Report results with and without off-support units".to_string(),
                ],
                diagnostics: Some(WarningDiagnostics {
                    name: "Fraction off-support".to_string(),
                    value: overlap.fraction_off_support,
                    threshold: Some(0.05),
                    exceeds_threshold: true,
                }),
            });
        }

        // 2. Extreme propensity scores (for IPW)
        if let Some(extreme) = overlap.extreme_weights {
            if extreme.max_weight > 20.0 {
                warnings.push(IdentificationWarning {
                    code: "EXTREME_WEIGHTS".to_string(),
                    severity: WarningSeverity::Warning,
                    title: "Extreme IPW Weights Detected".to_string(),
                    message: format!(
                        "Maximum IPW weight is {:.1}, indicating near-deterministic \
                         treatment assignment for some units. A few observations \
                         dominate the weighted estimate.",
                        extreme.max_weight
                    ),
                    assumption: "Positivity".to_string(),
                    remediation: vec![
                        "Trim or truncate weights".to_string(),
                        "Use stabilized weights".to_string(),
                        "Consider matching instead of IPW".to_string(),
                        "Report effective sample size".to_string(),
                    ],
                    diagnostics: Some(WarningDiagnostics {
                        name: "Maximum weight".to_string(),
                        value: extreme.max_weight,
                        threshold: Some(20.0),
                        exceeds_threshold: true,
                    }),
                });
            }
        }

        // 3. Covariate balance
        let balance = self.covariate_balance();
        let imbalanced_vars: Vec<_> = balance.iter()
            .filter(|b| b.std_diff.abs() > 0.1)
            .collect();

        if !imbalanced_vars.is_empty() {
            warnings.push(IdentificationWarning {
                code: "COVARIATE_IMBALANCE".to_string(),
                severity: WarningSeverity::Caution,
                title: "Residual Covariate Imbalance".to_string(),
                message: format!(
                    "{} covariates have standardized differences > 0.1 after \
                     matching/weighting: {}",
                    imbalanced_vars.len(),
                    imbalanced_vars.iter()
                        .take(5)
                        .map(|b| b.variable.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                assumption: "Covariate balance".to_string(),
                remediation: vec![
                    "Re-specify propensity score model".to_string(),
                    "Use doubly robust estimation".to_string(),
                    "Include imbalanced covariates in outcome model".to_string(),
                ],
                diagnostics: None,
            });
        }

        // 4. Unconfoundedness (always untestable)
        assumptions.push(Assumption {
            name: "Unconfoundedness (selection on observables)".to_string(),
            testable: false,
            description: "Treatment assignment is independent of potential outcomes conditional on observed covariates".to_string(),
            status: AssumptionStatus::Untestable,
        });

        // Add sensitivity analysis recommendation
        warnings.push(IdentificationWarning {
            code: "SENSITIVITY_RECOMMENDED".to_string(),
            severity: WarningSeverity::Info,
            title: "Consider Sensitivity Analysis".to_string(),
            message: "Unconfoundedness cannot be tested. Consider sensitivity analysis \
                      to assess robustness to unmeasured confounding.".to_string(),
            assumption: "Unconfoundedness".to_string(),
            remediation: vec![
                "Run sensemakr analysis for omitted variable bias bounds".to_string(),
                "Report E-values for unmeasured confounding".to_string(),
            ],
            diagnostics: None,
        });

        IdentificationReport {
            method: "Propensity Score Matching/IPW".to_string(),
            warnings,
            assumptions,
            has_critical: warnings.iter().any(|w| w.severity == WarningSeverity::Critical),
            summary: self.generate_summary(),
        }
    }
}
```

#### 3.4.5 Regression Discontinuity

```rust
// crates/p2a-core/src/diagnostics/generators/rd_warnings.rs

impl IdentificationCheck for RdResult {
    fn check_identification(&self, data: &Dataset) -> IdentificationReport {
        let mut warnings = Vec::new();
        let mut assumptions = Vec::new();

        // 1. McCrary manipulation test
        let mccrary = self.mccrary_test();
        assumptions.push(Assumption {
            name: "No manipulation".to_string(),
            testable: true,
            description: "Units cannot precisely manipulate the running variable to select into treatment".to_string(),
            status: if mccrary.p_value > 0.05 {
                AssumptionStatus::NoViolation
            } else {
                AssumptionStatus::LikelyViolation
            },
        });

        if mccrary.p_value < 0.05 {
            warnings.push(IdentificationWarning {
                code: "RD_MANIPULATION".to_string(),
                severity: WarningSeverity::Critical,
                title: "Running Variable Manipulation Detected".to_string(),
                message: format!(
                    "McCrary density test rejects continuity at the cutoff (p = {:.4}). \
                     There is a {:.1}% {} in density at the threshold, suggesting \
                     units may be manipulating their running variable to select into \
                     treatment.",
                    mccrary.p_value,
                    mccrary.discontinuity_percent.abs(),
                    if mccrary.discontinuity_percent > 0.0 { "increase" } else { "decrease" }
                ),
                assumption: "No manipulation".to_string(),
                remediation: vec![
                    "Examine the assignment mechanism for manipulation opportunities".to_string(),
                    "Consider donut-hole RD excluding units near cutoff".to_string(),
                    "Report McCrary plot and discuss".to_string(),
                    "If manipulation is one-sided, bounds may be available".to_string(),
                ],
                diagnostics: Some(WarningDiagnostics {
                    name: "McCrary p-value".to_string(),
                    value: mccrary.p_value,
                    threshold: Some(0.05),
                    exceeds_threshold: true,
                }),
            });
        }

        // 2. Covariate balance at cutoff
        if let Some(balance_test) = self.covariate_balance_test() {
            if balance_test.any_significant {
                warnings.push(IdentificationWarning {
                    code: "RD_COVARIATE_DISCONTINUITY".to_string(),
                    severity: WarningSeverity::Warning,
                    title: "Covariate Discontinuities at Cutoff".to_string(),
                    message: format!(
                        "Significant discontinuities detected in covariates at the \
                         cutoff: {}. This suggests either manipulation or confounding.",
                        balance_test.significant_vars.join(", ")
                    ),
                    assumption: "Local randomization".to_string(),
                    remediation: vec![
                        "Include discontinuous covariates as controls".to_string(),
                        "Investigate source of discontinuity".to_string(),
                    ],
                    diagnostics: None,
                });
            }
        }

        // 3. Bandwidth sensitivity
        if let Some(sensitivity) = self.bandwidth_sensitivity() {
            if sensitivity.coefficient_of_variation > 0.5 {
                warnings.push(IdentificationWarning {
                    code: "RD_BANDWIDTH_SENSITIVE".to_string(),
                    severity: WarningSeverity::Caution,
                    title: "Estimates Sensitive to Bandwidth".to_string(),
                    message: "Treatment effect estimates vary substantially across \
                              bandwidth choices. Results may be fragile.".to_string(),
                    assumption: "Continuity of conditional expectations".to_string(),
                    remediation: vec![
                        "Report estimates for multiple bandwidths".to_string(),
                        "Use robust bias-corrected inference (rdrobust)".to_string(),
                        "Examine local polynomial order sensitivity".to_string(),
                    ],
                    diagnostics: Some(WarningDiagnostics {
                        name: "CV across bandwidths".to_string(),
                        value: sensitivity.coefficient_of_variation,
                        threshold: Some(0.5),
                        exceeds_threshold: true,
                    }),
                });
            }
        }

        // 4. Sample size near cutoff
        let n_effective = self.effective_sample_size();
        if n_effective < 100 {
            warnings.push(IdentificationWarning {
                code: "RD_SMALL_SAMPLE".to_string(),
                severity: WarningSeverity::Caution,
                title: "Limited Observations Near Cutoff".to_string(),
                message: format!(
                    "Only {} observations within the optimal bandwidth. Local \
                     polynomial estimates may be imprecise.",
                    n_effective
                ),
                assumption: "Sufficient local data".to_string(),
                remediation: vec![
                    "Consider larger bandwidth (with bias correction)".to_string(),
                    "Report confidence intervals prominently".to_string(),
                ],
                diagnostics: None,
            });
        }

        IdentificationReport {
            method: "Regression Discontinuity".to_string(),
            warnings,
            assumptions,
            has_critical: warnings.iter().any(|w| w.severity == WarningSeverity::Critical),
            summary: self.generate_summary(),
        }
    }
}
```

---

## 4. MCP Integration

### 4.1 Tool Response Format

```rust
// crates/p2a-mcp/src/responses.rs

#[derive(Serialize)]
pub struct CausalMethodResponse {
    /// Standard estimation results
    pub results: EstimationResults,
    /// Identification report (new)
    pub identification: IdentificationReport,
    /// Whether to prominently display warnings
    pub show_warnings: bool,
}
```

### 4.2 LLM Prompt Engineering

The MCP server includes identification information in tool responses. The LLM system prompt should instruct:

```
When reporting results from causal inference methods (DiD, IV, RD, matching),
you MUST:

1. Report any warnings with severity "Warning" or "Critical" BEFORE presenting
   coefficient estimates.

2. For warnings with severity "Critical", explicitly state that results may be
   unreliable and explain why.

3. List the key assumptions required for causal interpretation and their
   testable status.

4. Use associational language ("is associated with") rather than causal
   language ("causes", "effect of") UNLESS the identification report shows
   no warnings and all testable assumptions pass.

5. When users request "causal effects" or "treatment effects", ask about
   their identification strategy if not already specified.
```

### 4.3 User Prompt Detection

The LLM layer should detect causal language in user prompts:

```rust
// crates/p2a-mcp/src/prompt_analysis.rs

/// Patterns indicating causal intent
const CAUSAL_PATTERNS: &[&str] = &[
    "effect of",
    "impact of",
    "causal",
    "treatment effect",
    "caused by",
    "leads to",
    "results in",
    "ATT",
    "ATE",
    "LATE",
];

/// Check if prompt suggests causal intent
pub fn has_causal_intent(prompt: &str) -> bool {
    let prompt_lower = prompt.to_lowercase();
    CAUSAL_PATTERNS.iter().any(|p| prompt_lower.contains(p))
}

/// Generate clarifying question for causal requests
pub fn causal_clarification_prompt(method: &str) -> String {
    format!(
        "You've requested a {} analysis using causal language. To ensure \
         appropriate interpretation, could you briefly describe:\n\
         1. What is your identification strategy?\n\
         2. Why do you believe the key assumptions hold in your context?\n\n\
         I'll proceed with the analysis either way, but this helps me provide \
         appropriate caveats.",
        method
    )
}
```

---

## 5. User Interface

### 5.1 Warning Display Format

Warnings are formatted for LLM output as structured blocks:

```markdown
⚠️ **IDENTIFICATION WARNING: Weak Instrument Detected**

First-stage F-statistic is 6.2, below the threshold of 10.

**Assumption violated:** Instrument relevance

**Why this matters:** Weak instruments cause bias toward OLS estimates and
unreliable inference. Standard errors may substantially understate uncertainty.

**Recommendations:**
- Consider stronger instruments
- Use weak-instrument-robust inference (Anderson-Rubin test)
- Report reduced-form estimates as robustness check

---
```

### 5.2 Severity Icons

| Severity | Icon | Meaning |
|----------|------|---------|
| Info | ℹ️ | Assumption stated, no action needed |
| Caution | ⚡ | Some concern, consider robustness |
| Warning | ⚠️ | Potential violation, interpret carefully |
| Critical | 🚨 | Likely violation, results may be invalid |

### 5.3 Configuration Options

Users can configure warning behavior:

```rust
#[derive(Deserialize)]
pub struct WarningConfig {
    /// Minimum severity to display
    pub min_severity: WarningSeverity,
    /// Whether to show remediation suggestions
    pub show_remediation: bool,
    /// Whether to include diagnostic values
    pub show_diagnostics: bool,
    /// Whether LLM should ask clarifying questions
    pub ask_clarifications: bool,
}

impl Default for WarningConfig {
    fn default() -> Self {
        Self {
            min_severity: WarningSeverity::Caution,
            show_remediation: true,
            show_diagnostics: true,
            ask_clarifications: true,
        }
    }
}
```

---

## 6. Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

1. Define data structures (`IdentificationWarning`, `IdentificationReport`)
2. Create `IdentificationCheck` trait
3. Implement IV warnings (weak instruments, overidentification)
4. Add to MCP tool responses

### Phase 2: DiD and Matching (Week 3-4)

1. Implement DiD warnings (parallel trends, anticipation)
2. Implement matching/IPW warnings (overlap, balance, extreme weights)
3. Add SUTVA check infrastructure

### Phase 3: RD and Cross-Method (Week 5-6)

1. Implement RD warnings (McCrary, bandwidth sensitivity)
2. Implement geographic SUTVA checks
3. Add LLM prompt detection for causal language

### Phase 4: Polish and Documentation (Week 7-8)

1. User-facing documentation
2. Configuration options
3. Testing and validation
4. Paper update (Section 2.3 expansion)

---

## 7. Testing Strategy

### 7.1 Unit Tests

Each warning generator needs tests for:
- Correct detection of violations
- Appropriate severity assignment
- Edge cases (missing data, boundary values)

### 7.2 Integration Tests

Test full pipeline from estimation through warning display:

```rust
#[test]
fn test_weak_instrument_warning_displayed() {
    // Generate data with weak instrument
    let data = generate_weak_iv_data(n = 1000, first_stage_r2 = 0.01);

    // Run 2SLS
    let result = run_2sls(&data, "y", "x_endog", &["z_weak"], &[]);

    // Check warning generated
    let report = result.check_identification(&data);
    assert!(report.warnings.iter().any(|w| w.code == "WEAK_INSTRUMENT"));
    assert!(report.has_critical || report.warnings.iter().any(|w|
        w.severity == WarningSeverity::Warning));
}
```

### 7.3 Validation Against Known Cases

Use published datasets with known identification issues:
- Weak IV: Angrist-Krueger quarter-of-birth
- Parallel trends violation: Constructed examples
- RD manipulation: Lee (2008) close elections (manipulation at 0)

---

## 8. Open Questions

1. **Should warnings block execution?** Current design is non-blocking. Should "Critical" warnings require user confirmation?

2. **How aggressive should SUTVA heuristics be?** Geographic proximity thresholds are arbitrary. False positives may annoy users.

3. **LLM clarification questions:** Should the LLM always ask about identification for causal requests, or only when specific risk factors are detected?

4. **Storing warnings:** Should identification reports be persisted for audit trails?

5. **Custom thresholds:** Should users be able to configure thresholds (e.g., F > 10, SMD < 0.1)?

---

## 9. References

- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression.
- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design.
- Imbens, G. W., & Rubin, D. B. (2015). Causal inference for statistics, social, and biomedical sciences.
- Cattaneo, M. D., Idrobo, N., & Titiunik, R. (2020). A practical introduction to regression discontinuity designs.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods.
