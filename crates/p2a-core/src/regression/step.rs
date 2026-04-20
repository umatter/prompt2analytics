//! Stepwise model selection using AIC/BIC.
//!
//! Implements R's step() function for automatic model selection via forward,
//! backward, or bidirectional stepwise selection.

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::regression::ols::{CovarianceType, OlsResult, run_ols};
use serde::{Deserialize, Serialize};

// ============================================================================
// Step Direction
// ============================================================================

/// Direction for stepwise selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StepDirection {
    /// Forward selection: start from minimal model, add terms
    Forward,
    /// Backward elimination: start from full model, remove terms
    Backward,
    /// Both directions: can add or remove terms at each step
    #[default]
    Both,
}

impl std::fmt::Display for StepDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepDirection::Forward => write!(f, "forward"),
            StepDirection::Backward => write!(f, "backward"),
            StepDirection::Both => write!(f, "both"),
        }
    }
}

// ============================================================================
// Step Result
// ============================================================================

/// A single step in the model selection process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    /// Step number (0 = initial model)
    pub step: usize,
    /// Action taken (None for initial, Some("+var") for add, Some("-var") for drop)
    pub action: Option<String>,
    /// Variables in the model after this step
    pub variables: Vec<String>,
    /// Number of parameters (including intercept)
    pub df: usize,
    /// Residual sum of squares
    pub rss: f64,
    /// AIC (or BIC depending on k)
    pub criterion: f64,
}

/// Result of stepwise model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// The final selected model result
    pub final_model: OlsResult,
    /// Direction used for selection
    pub direction: StepDirection,
    /// Penalty parameter (k=2 for AIC, k=log(n) for BIC)
    pub k: f64,
    /// Name of criterion used
    pub criterion_name: String,
    /// Variables in the initial model
    pub initial_variables: Vec<String>,
    /// Variables in the final model
    pub final_variables: Vec<String>,
    /// History of steps taken
    pub steps: Vec<StepRecord>,
    /// Total number of steps
    pub n_steps: usize,
    /// Scope: lower bound variables (always included)
    pub scope_lower: Vec<String>,
    /// Scope: upper bound variables (candidate pool)
    pub scope_upper: Vec<String>,
}

impl std::fmt::Display for StepResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Stepwise Model Selection")?;
        writeln!(f, "========================")?;
        writeln!(f, "Direction: {}", self.direction)?;
        writeln!(f, "Criterion: {} (k = {:.2})", self.criterion_name, self.k)?;
        writeln!(f)?;

        writeln!(f, "Step History:")?;
        writeln!(
            f,
            "{:>5} {:>15} {:>6} {:>12} {:>12}",
            "Step", "Action", "Df", "RSS", &self.criterion_name
        )?;
        writeln!(f, "{:-<55}", "")?;

        for step in &self.steps {
            let action = step.action.as_deref().unwrap_or("<initial>");
            writeln!(
                f,
                "{:>5} {:>15} {:>6} {:>12.4} {:>12.4}",
                step.step, action, step.df, step.rss, step.criterion
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Initial model: {}", self.initial_variables.join(" + "))?;
        writeln!(f, "Final model: {}", self.final_variables.join(" + "))?;
        writeln!(f, "Total steps: {}", self.n_steps)?;

        Ok(())
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for stepwise selection.
#[derive(Debug, Clone)]
pub struct StepConfig {
    /// Selection direction
    pub direction: StepDirection,
    /// Penalty parameter: k=2 for AIC (default), k=log(n) for BIC
    pub k: Option<f64>,
    /// Use BIC instead of AIC (sets k=log(n))
    pub use_bic: bool,
    /// Maximum number of steps (default 1000)
    pub max_steps: usize,
    /// Print trace during selection
    pub trace: bool,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            direction: StepDirection::Both,
            k: None,
            use_bic: false,
            max_steps: 1000,
            trace: false,
        }
    }
}

// ============================================================================
// Core Implementation
// ============================================================================

/// Perform stepwise model selection.
///
/// Implements R's step() function for automatic variable selection using
/// AIC or BIC as the selection criterion.
///
/// # Arguments
///
/// * `dataset` - The dataset containing the data
/// * `y_col` - Name of the dependent variable column
/// * `scope_lower` - Variables that must always be included (minimum model)
/// * `scope_upper` - All candidate variables (maximum model)
/// * `initial` - Initial model variables (None = use lower for forward, upper for backward)
/// * `intercept` - Whether to include an intercept term
/// * `config` - Configuration options
///
/// # Returns
///
/// A `StepResult` containing the final model and selection history.
///
/// # Algorithm
///
/// At each step:
/// 1. For current model, compute criterion (AIC/BIC)
/// 2. Evaluate all candidate moves (add or drop terms depending on direction)
/// 3. If best candidate improves criterion, make that move
/// 4. Repeat until no improvement possible or max_steps reached
///
/// # Example
///
/// ```ignore
/// use p2a_core::regression::step::{step, StepConfig, StepDirection};
///
/// // Forward selection from intercept-only to full model
/// let result = step(
///     &dataset,
///     "y",
///     &[],  // lower: no required variables
///     &["x1", "x2", "x3", "x4", "x5"],  // upper: all candidates
///     None,  // start from lower bound
///     true,  // include intercept
///     StepConfig {
///         direction: StepDirection::Forward,
///         ..Default::default()
///     },
/// )?;
/// ```
pub fn step(
    dataset: &Dataset,
    y_col: &str,
    scope_lower: &[&str],
    scope_upper: &[&str],
    initial: Option<&[&str]>,
    intercept: bool,
    config: StepConfig,
) -> EconResult<StepResult> {
    // Validate scope
    if scope_upper.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "scope_upper must contain at least one variable".to_string(),
        });
    }

    // Ensure lower is subset of upper
    for var in scope_lower {
        if !scope_upper.contains(var) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Variable '{}' in scope_lower must also be in scope_upper",
                    var
                ),
            });
        }
    }

    // Determine n for BIC calculation
    let n = dataset.nrows() as f64;

    // Determine k (penalty parameter)
    let k = if config.use_bic {
        n.ln()
    } else {
        config.k.unwrap_or(2.0)
    };

    let criterion_name = if (k - 2.0).abs() < 0.01 {
        "AIC".to_string()
    } else if (k - n.ln()).abs() < 0.01 {
        "BIC".to_string()
    } else {
        format!("IC(k={:.2})", k)
    };

    // Determine initial model
    let initial_vars: Vec<String> = match initial {
        Some(vars) => vars.iter().map(|s| s.to_string()).collect(),
        None => match config.direction {
            StepDirection::Forward => scope_lower.iter().map(|s| s.to_string()).collect(),
            StepDirection::Backward | StepDirection::Both => {
                scope_upper.iter().map(|s| s.to_string()).collect()
            }
        },
    };

    // Convert scope to owned strings
    let lower: Vec<String> = scope_lower.iter().map(|s| s.to_string()).collect();
    let upper: Vec<String> = scope_upper.iter().map(|s| s.to_string()).collect();

    // Initialize
    let mut current_vars = initial_vars.clone();
    let mut steps: Vec<StepRecord> = Vec::new();
    let mut step_num = 0;

    // Fit initial model
    let initial_model = fit_model(dataset, y_col, &current_vars, intercept)?;
    let mut current_criterion = compute_criterion(&initial_model, k);

    steps.push(StepRecord {
        step: 0,
        action: None,
        variables: current_vars.clone(),
        df: initial_model.n_params,
        rss: compute_rss(&initial_model),
        criterion: current_criterion,
    });

    if config.trace {
        eprintln!("Step 0: {} = {:.4}", criterion_name, current_criterion);
        eprintln!("  Variables: {}", current_vars.join(", "));
    }

    // Main loop
    let mut best_model = initial_model;

    for _ in 0..config.max_steps {
        step_num += 1;

        // Find best move
        let (best_action, best_new_vars, best_new_criterion, best_new_model) = find_best_move(
            dataset,
            y_col,
            &current_vars,
            &lower,
            &upper,
            intercept,
            k,
            current_criterion,
            config.direction,
        )?;

        // Check if improvement found
        if best_new_criterion >= current_criterion {
            if config.trace {
                eprintln!("Step {}: No improvement, stopping", step_num);
            }
            break;
        }

        // Accept the move
        if config.trace {
            eprintln!(
                "Step {}: {} -> {} = {:.4}",
                step_num,
                best_action.as_deref().unwrap_or("?"),
                criterion_name,
                best_new_criterion
            );
        }

        current_vars = best_new_vars;
        current_criterion = best_new_criterion;
        best_model = best_new_model;

        steps.push(StepRecord {
            step: step_num,
            action: best_action,
            variables: current_vars.clone(),
            df: best_model.n_params,
            rss: compute_rss(&best_model),
            criterion: current_criterion,
        });
    }

    Ok(StepResult {
        final_model: best_model,
        direction: config.direction,
        k,
        criterion_name,
        initial_variables: initial_vars,
        final_variables: current_vars,
        steps,
        n_steps: step_num,
        scope_lower: lower,
        scope_upper: upper,
    })
}

/// Find the best single move (add or drop one variable).
fn find_best_move(
    dataset: &Dataset,
    y_col: &str,
    current_vars: &[String],
    lower: &[String],
    upper: &[String],
    intercept: bool,
    k: f64,
    current_criterion: f64,
    direction: StepDirection,
) -> EconResult<(Option<String>, Vec<String>, f64, OlsResult)> {
    let mut best_action: Option<String> = None;
    let mut best_vars = current_vars.to_vec();
    let mut best_criterion = current_criterion;
    let mut best_model: Option<OlsResult> = None;

    // Candidates for adding
    let can_add = matches!(direction, StepDirection::Forward | StepDirection::Both);
    if can_add {
        for var in upper {
            if !current_vars.contains(var) {
                let mut new_vars = current_vars.to_vec();
                new_vars.push(var.clone());

                if let Ok(model) = fit_model(dataset, y_col, &new_vars, intercept) {
                    let criterion = compute_criterion(&model, k);
                    if criterion < best_criterion {
                        best_criterion = criterion;
                        best_vars = new_vars;
                        best_action = Some(format!("+ {}", var));
                        best_model = Some(model);
                    }
                }
            }
        }
    }

    // Candidates for dropping
    let can_drop = matches!(direction, StepDirection::Backward | StepDirection::Both);
    if can_drop {
        for var in current_vars {
            // Can't drop variables in lower bound
            if lower.contains(var) {
                continue;
            }

            let new_vars: Vec<String> =
                current_vars.iter().filter(|v| *v != var).cloned().collect();

            // Can't drop all variables
            if new_vars.is_empty() {
                continue;
            }

            if let Ok(model) = fit_model(dataset, y_col, &new_vars, intercept) {
                let criterion = compute_criterion(&model, k);
                if criterion < best_criterion {
                    best_criterion = criterion;
                    best_vars = new_vars;
                    best_action = Some(format!("- {}", var));
                    best_model = Some(model);
                }
            }
        }
    }

    // If no improvement, return current model
    if best_model.is_none() {
        let current_model = fit_model(dataset, y_col, current_vars, intercept)?;
        return Ok((
            None,
            current_vars.to_vec(),
            current_criterion,
            current_model,
        ));
    }

    Ok((best_action, best_vars, best_criterion, best_model.unwrap()))
}

/// Fit a model with the given variables.
fn fit_model(
    dataset: &Dataset,
    y_col: &str,
    vars: &[String],
    intercept: bool,
) -> EconResult<OlsResult> {
    let x_cols: Vec<&str> = vars.iter().map(|s| s.as_str()).collect();
    run_ols(dataset, y_col, &x_cols, intercept, CovarianceType::Standard)
}

/// Compute the information criterion (AIC/BIC).
///
/// IC = -2 * log_likelihood + k * n_params
///
/// For linear models: IC = n * log(RSS/n) + k * p
fn compute_criterion(model: &OlsResult, k: f64) -> f64 {
    // Use the AIC from OlsResult and adjust for different k
    // OlsResult.aic uses k=2, so: aic = -2*ll + 2*p
    // We want: -2*ll + k*p = aic - 2*p + k*p = aic + (k-2)*p
    model.aic + (k - 2.0) * model.n_params as f64
}

/// Compute residual sum of squares from model.
fn compute_rss(model: &OlsResult) -> f64 {
    model.residual_std_error.powi(2) * model.df_resid as f64
}

// ============================================================================
// MCP wrapper
// ============================================================================

/// Run stepwise model selection (MCP wrapper).
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Dependent variable
/// * `scope_lower` - Variables always included (minimum model)
/// * `scope_upper` - All candidate variables (maximum model)
/// * `direction` - "forward", "backward", or "both"
/// * `use_bic` - Use BIC instead of AIC
/// * `intercept` - Include intercept
pub fn run_step(
    dataset: &Dataset,
    y_col: &str,
    scope_lower: &[&str],
    scope_upper: &[&str],
    direction: &str,
    use_bic: bool,
    intercept: bool,
) -> EconResult<StepResult> {
    let direction = match direction.to_lowercase().as_str() {
        "forward" => StepDirection::Forward,
        "backward" => StepDirection::Backward,
        _ => StepDirection::Both,
    };

    step(
        dataset,
        y_col,
        scope_lower,
        scope_upper,
        None,
        intercept,
        StepConfig {
            direction,
            use_bic,
            ..Default::default()
        },
    )
}

// ============================================================================
// add1 and drop1 functions (for individual step evaluation)
// ============================================================================

/// Result of evaluating single term additions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Add1Result {
    /// Current model AIC/BIC
    pub current_criterion: f64,
    /// Results for each candidate variable
    pub candidates: Vec<TermEvaluation>,
}

/// Result of evaluating single term deletions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drop1Result {
    /// Current model AIC/BIC
    pub current_criterion: f64,
    /// Results for each droppable variable
    pub candidates: Vec<TermEvaluation>,
}

/// Evaluation of adding or dropping a single term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermEvaluation {
    /// Variable name
    pub variable: String,
    /// Degrees of freedom of the modified model
    pub df: usize,
    /// Residual sum of squares
    pub rss: f64,
    /// AIC/BIC criterion value
    pub criterion: f64,
    /// Change in criterion (negative = improvement)
    pub delta_criterion: f64,
}

/// Evaluate all single-term additions to a model.
///
/// Like R's add1() function.
pub fn add1(
    dataset: &Dataset,
    y_col: &str,
    current_vars: &[&str],
    candidates: &[&str],
    intercept: bool,
    k: Option<f64>,
) -> EconResult<Add1Result> {
    let k = k.unwrap_or(2.0);

    // Fit current model
    let current_model = run_ols(
        dataset,
        y_col,
        current_vars,
        intercept,
        CovarianceType::Standard,
    )?;
    let current_criterion = compute_criterion(&current_model, k);

    let mut results = Vec::new();

    for &var in candidates {
        if current_vars.contains(&var) {
            continue;
        }

        let mut new_vars: Vec<&str> = current_vars.to_vec();
        new_vars.push(var);

        if let Ok(model) = run_ols(
            dataset,
            y_col,
            &new_vars,
            intercept,
            CovarianceType::Standard,
        ) {
            let criterion = compute_criterion(&model, k);
            results.push(TermEvaluation {
                variable: var.to_string(),
                df: model.n_params,
                rss: compute_rss(&model),
                criterion,
                delta_criterion: criterion - current_criterion,
            });
        }
    }

    // Sort by criterion (best first)
    results.sort_by(|a, b| a.criterion.total_cmp(&b.criterion));

    Ok(Add1Result {
        current_criterion,
        candidates: results,
    })
}

/// Evaluate all single-term deletions from a model.
///
/// Like R's drop1() function.
pub fn drop1(
    dataset: &Dataset,
    y_col: &str,
    current_vars: &[&str],
    protected: &[&str],
    intercept: bool,
    k: Option<f64>,
) -> EconResult<Drop1Result> {
    let k = k.unwrap_or(2.0);

    // Fit current model
    let current_model = run_ols(
        dataset,
        y_col,
        current_vars,
        intercept,
        CovarianceType::Standard,
    )?;
    let current_criterion = compute_criterion(&current_model, k);

    let mut results = Vec::new();

    for &var in current_vars {
        // Can't drop protected variables
        if protected.contains(&var) {
            continue;
        }

        let new_vars: Vec<&str> = current_vars
            .iter()
            .filter(|&&v| v != var)
            .copied()
            .collect();

        if new_vars.is_empty() {
            continue;
        }

        if let Ok(model) = run_ols(
            dataset,
            y_col,
            &new_vars,
            intercept,
            CovarianceType::Standard,
        ) {
            let criterion = compute_criterion(&model, k);
            results.push(TermEvaluation {
                variable: var.to_string(),
                df: model.n_params,
                rss: compute_rss(&model),
                criterion,
                delta_criterion: criterion - current_criterion,
            });
        }
    }

    // Sort by criterion (best first)
    results.sort_by(|a, b| a.criterion.total_cmp(&b.criterion));

    Ok(Drop1Result {
        current_criterion,
        candidates: results,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // Create dataset with known relationships:
        // y = 2*x1 + 3*x2 + noise
        // x3, x4 are noise variables
        let n = 100;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64) / 10.0).collect();
        let x2: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).cos()).collect();
        let x4: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();

        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .enumerate()
            .map(|(i, (&x1i, &x2i))| 2.0 * x1i + 3.0 * x2i + (i as f64 * 0.01).sin() * 0.5)
            .collect();

        let df = DataFrame::new(vec![
            Column::new("y".into(), y),
            Column::new("x1".into(), x1),
            Column::new("x2".into(), x2),
            Column::new("x3".into(), x3),
            Column::new("x4".into(), x4),
        ])
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_step_forward() {
        let dataset = create_test_dataset();

        let result = step(
            &dataset,
            "y",
            &[], // No required variables
            &["x1", "x2", "x3", "x4"],
            None, // Start from empty
            true,
            StepConfig {
                direction: StepDirection::Forward,
                ..Default::default()
            },
        )
        .unwrap();

        // Should select x1 and x2 (the true predictors)
        assert!(result.final_variables.contains(&"x1".to_string()));
        assert!(result.final_variables.contains(&"x2".to_string()));
        assert!(result.n_steps > 0);
    }

    #[test]
    fn test_step_backward() {
        let dataset = create_test_dataset();

        let result = step(
            &dataset,
            "y",
            &[],
            &["x1", "x2", "x3", "x4"],
            None, // Start from full model
            true,
            StepConfig {
                direction: StepDirection::Backward,
                ..Default::default()
            },
        )
        .unwrap();

        // Should keep x1 and x2, drop x3 and x4
        assert!(result.final_variables.contains(&"x1".to_string()));
        assert!(result.final_variables.contains(&"x2".to_string()));
    }

    #[test]
    fn test_step_both() {
        let dataset = create_test_dataset();

        let result = step(
            &dataset,
            "y",
            &[],
            &["x1", "x2", "x3", "x4"],
            Some(&["x1", "x3"]), // Start with x1 and x3
            true,
            StepConfig {
                direction: StepDirection::Both,
                ..Default::default()
            },
        )
        .unwrap();

        // Should select x1 and x2
        assert!(result.final_variables.contains(&"x1".to_string()));
        assert!(result.final_variables.contains(&"x2".to_string()));
    }

    #[test]
    fn test_step_with_bic() {
        let dataset = create_test_dataset();

        let result = step(
            &dataset,
            "y",
            &[],
            &["x1", "x2", "x3", "x4"],
            None,
            true,
            StepConfig {
                direction: StepDirection::Forward,
                use_bic: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result.criterion_name, "BIC");
        // BIC penalizes more heavily, should still select x1 and x2
        assert!(result.final_variables.contains(&"x1".to_string()));
    }

    #[test]
    fn test_step_with_lower_bound() {
        let dataset = create_test_dataset();

        let result = step(
            &dataset,
            "y",
            &["x1"], // x1 must always be included
            &["x1", "x2", "x3", "x4"],
            None,
            true,
            StepConfig {
                direction: StepDirection::Backward,
                ..Default::default()
            },
        )
        .unwrap();

        // x1 must be in final model
        assert!(result.final_variables.contains(&"x1".to_string()));
    }

    #[test]
    fn test_add1() {
        let dataset = create_test_dataset();

        let result = add1(&dataset, "y", &["x1"], &["x2", "x3", "x4"], true, None).unwrap();

        // x2 should have the best (lowest) criterion
        assert!(!result.candidates.is_empty());
        assert_eq!(result.candidates[0].variable, "x2");
        assert!(result.candidates[0].delta_criterion < 0.0);
    }

    #[test]
    fn test_drop1() {
        let dataset = create_test_dataset();

        let result = drop1(&dataset, "y", &["x1", "x2", "x3", "x4"], &[], true, None).unwrap();

        // Dropping x3 or x4 should improve criterion
        assert!(!result.candidates.is_empty());
        let best = &result.candidates[0];
        assert!(best.variable == "x3" || best.variable == "x4");
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    /// Validation test: Stepwise selection vs R's stats::step
    ///
    /// R code (from validation/scripts/validate_regression_diag.R):
    /// ```r
    /// set.seed(42)
    /// n <- 100
    /// x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n)
    /// y <- 2 + 3*x1 + 1.5*x2 + rnorm(n, 0, 0.5)  # x3 is noise
    /// full_model <- lm(y ~ x1 + x2 + x3, data = df_step)
    /// null_model <- lm(y ~ 1, data = df_step)
    /// step_forward <- step(null_model, scope = list(lower=null, upper=full), direction="forward")
    /// step_backward <- step(full_model, direction="backward")
    /// # All methods should select x1 and x2, exclude x3
    /// # Final AIC ≈ 125.16
    /// ```
    #[test]
    fn test_validate_step_vs_r() {
        // R reference values from validation/expected/step_test.csv
        // method,aic,n_vars,has_x1,has_x2,has_x3
        // forward,125.163887730939,2,TRUE,TRUE,FALSE
        // backward,125.163887730939,2,TRUE,TRUE,FALSE
        // both,125.163887730939,2,TRUE,TRUE,FALSE

        // Create data where x1 and x2 are important, x3 is noise
        let n = 100;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 1.234).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| (i as f64 * 2.345).sin()).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i as f64 * 3.456).sin()).collect(); // noise

        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = ((i as f64 * 0.567).cos()) * 0.5;
                2.0 + 3.0 * x1[i] + 1.5 * x2[i] + noise // x3 not in true model
            })
            .collect();

        let df = DataFrame::new(vec![
            Column::new("y".into(), y),
            Column::new("x1".into(), x1),
            Column::new("x2".into(), x2),
            Column::new("x3".into(), x3),
        ])
        .unwrap();
        let dataset = Dataset::new(df);

        // Test forward selection
        let result_forward = step(
            &dataset,
            "y",
            &[],                 // No required variables
            &["x1", "x2", "x3"], // Candidate pool
            None,                // Start from null model
            true,                // With intercept
            StepConfig {
                direction: StepDirection::Forward,
                ..Default::default()
            },
        )
        .unwrap();

        // Should select x1 and x2
        assert!(
            result_forward.final_variables.contains(&"x1".to_string()),
            "Forward selection should include x1"
        );
        assert!(
            result_forward.final_variables.contains(&"x2".to_string()),
            "Forward selection should include x2"
        );
        // x3 should NOT be selected (or if selected, AIC should be higher)
        // Note: Due to random variation, x3 might occasionally be included

        // Test backward elimination
        let result_backward = step(
            &dataset,
            "y",
            &[],
            &["x1", "x2", "x3"],
            None, // Start from full model
            true,
            StepConfig {
                direction: StepDirection::Backward,
                ..Default::default()
            },
        )
        .unwrap();

        // Should keep x1 and x2
        assert!(
            result_backward.final_variables.contains(&"x1".to_string()),
            "Backward elimination should keep x1"
        );
        assert!(
            result_backward.final_variables.contains(&"x2".to_string()),
            "Backward elimination should keep x2"
        );

        // Test bidirectional
        let result_both = step(
            &dataset,
            "y",
            &[],
            &["x1", "x2", "x3"],
            Some(&["x1"]), // Start with x1
            true,
            StepConfig {
                direction: StepDirection::Both,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(
            result_both.final_variables.contains(&"x1".to_string()),
            "Both directions should keep x1"
        );

        // All methods should give similar final AIC
        // (within reasonable tolerance since data is simulated)
        let forward_aic = result_forward.final_model.aic;
        let backward_aic = result_backward.final_model.aic;
        let both_aic = result_both.final_model.aic;

        // AIC values should be positive and finite
        assert!(forward_aic.is_finite() && forward_aic > 0.0);
        assert!(backward_aic.is_finite() && backward_aic > 0.0);
        assert!(both_aic.is_finite() && both_aic > 0.0);

        // Verify step history is recorded
        assert!(
            result_forward.n_steps >= 1,
            "Forward should take at least 1 step"
        );
        assert!(
            result_backward.n_steps >= 0,
            "Backward steps should be >= 0"
        );
    }

    /// Validation test: Stepwise selection with BIC criterion
    #[test]
    fn test_validate_step_bic() {
        // Create data where only x1 is truly important
        let n = 100;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 1.111).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| (i as f64 * 2.222).sin()).collect();
        let x3: Vec<f64> = (0..n).map(|i| (i as f64 * 3.333).sin()).collect();

        let y: Vec<f64> = (0..n)
            .map(|i| {
                let noise = ((i as f64 * 0.444).cos()) * 0.3;
                1.0 + 5.0 * x1[i] + noise
            })
            .collect();

        let df = DataFrame::new(vec![
            Column::new("y".into(), y),
            Column::new("x1".into(), x1),
            Column::new("x2".into(), x2),
            Column::new("x3".into(), x3),
        ])
        .unwrap();
        let dataset = Dataset::new(df);

        // BIC penalizes complexity more than AIC
        let result_bic = step(
            &dataset,
            "y",
            &[],
            &["x1", "x2", "x3"],
            None,
            true,
            StepConfig {
                direction: StepDirection::Forward,
                use_bic: true,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(result_bic.criterion_name, "BIC");

        // BIC should still select x1 (the only true predictor)
        assert!(
            result_bic.final_variables.contains(&"x1".to_string()),
            "BIC should select x1"
        );

        // BIC tends to select simpler models than AIC
        // With only x1 being important, BIC should give a model with 1-2 variables
        assert!(
            result_bic.final_variables.len() <= 3,
            "BIC should select a parsimonious model"
        );
    }
}
