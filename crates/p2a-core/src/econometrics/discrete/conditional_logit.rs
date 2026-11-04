//! McFadden's Conditional Logit (mlogit).
//!
//! # Mathematical Background
//!
//! McFadden's conditional logit (also called "mixed logit" in some contexts) specifies:
//!
//! U_ij = V_ij + epsilon_ij
//!
//! where:
//! - V_ij = X_ij'beta + Z_i'gamma_j is the deterministic utility
//! - X_ij are alternative-specific variables
//! - Z_i are individual-specific variables
//! - beta are generic coefficients (same across all alternatives)
//! - gamma_j are alternative-specific coefficients
//!
//! By default the model includes J - 1 alternative-specific constants (a constant for
//! every alternative but the reference, the first in factor order unless a reference is
//! given), as `mlogit::mlogit(choice ~ x | z)` does; [`MlogitConfig::asc`] = `false` fits
//! `mlogit(choice ~ x | 0 + z)`, which for generic variables only is
//! `survival::clogit`'s model. Fitted by Newton's method on the centred information
//! (equilibrated, Cholesky first), stopped by the Newton decrement on `glm.fit`'s
//! relative-deviance scale.
//!
//! # References
//!
//! - McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
//! - Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).
//!
//! R equivalent: `mlogit::mlogit()`

use faer::linalg::solvers::Solve;
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::ndarray_to_faer;
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, sym_inverse};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, deviance_change, notes_text};
use crate::traits::estimator::normal_p_value_two_sided;

use super::newton::{MAX_HALVINGS, ensure_dense_inverse_fits};

/// Result from McFadden's conditional logit model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlogitResult {
    /// Alternative-specific variable names (generic coefficients)
    pub alt_specific_vars: Vec<String>,
    /// Individual-specific variable names (alternative-specific coefficients)
    pub ind_specific_vars: Vec<String>,
    /// Alternative names
    pub alternatives: Vec<String>,
    /// Reference alternative
    pub reference_alternative: String,
    /// Generic coefficients for alternative-specific variables
    pub beta: Vec<f64>,
    /// Standard errors for beta
    pub beta_std_errors: Vec<f64>,
    /// Z-statistics for beta
    pub beta_z_stats: Vec<f64>,
    /// P-values for beta
    pub beta_p_values: Vec<f64>,
    /// Alternative-specific coefficients for individual-specific variables
    pub gamma: Vec<Vec<f64>>,
    /// Standard errors for gamma
    pub gamma_std_errors: Vec<Vec<f64>>,
    /// Z-statistics for gamma
    pub gamma_z_stats: Vec<Vec<f64>>,
    /// P-values for gamma
    pub gamma_p_values: Vec<Vec<f64>>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Log-likelihood of null model
    pub log_likelihood_null: f64,
    /// McFadden's pseudo R-squared
    pub pseudo_r_squared: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Number of choice situations
    pub n_choice_situations: usize,
    /// Number of alternatives
    pub n_alternatives: usize,
    /// Choice counts by alternative
    pub choice_counts: Vec<usize>,
    /// The model has J - 1 alternative-specific constants (`ind_specific_vars[0]` is
    /// `(Intercept)`; `gamma[j][0]` is the constant of the j-th non-reference alternative).
    #[serde(default)]
    pub asc: bool,
    /// Some fitted choice probability is numerically 0 or 1 (within `10 eps` of the bound).
    #[serde(default)]
    pub separation: bool,
    /// Convergence record of the Newton iteration.
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings (convergence, separation, singular information).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl MlogitResult {
    /// The non-reference alternatives, in `alternatives` order (the order of `gamma`).
    fn non_reference(&self) -> impl Iterator<Item = &String> {
        self.alternatives
            .iter()
            .filter(move |a| **a != self.reference_alternative)
    }

    /// Whether `ind_specific_vars[0]` holds the constants.
    fn has_constants(&self) -> bool {
        self.asc
            && self
                .ind_specific_vars
                .first()
                .is_some_and(|v| v == "(Intercept)")
    }

    /// mlogit's coefficient names and order: `(Intercept):alt` for the non-reference
    /// alternatives (with constants), the generic variables, then `z:alt` variable by
    /// variable.
    pub fn coefficient_names(&self) -> Vec<String> {
        let skip = usize::from(self.has_constants());
        let mut names = Vec::new();
        if skip == 1 {
            names.extend(self.non_reference().map(|a| format!("(Intercept):{a}")));
        }
        names.extend(self.alt_specific_vars.iter().cloned());
        for v in self.ind_specific_vars.iter().skip(skip) {
            names.extend(self.non_reference().map(|a| format!("{v}:{a}")));
        }
        names
    }

    fn in_mlogit_order(&self, beta: &[f64], gamma: &[Vec<f64>]) -> Vec<f64> {
        let skip = usize::from(self.has_constants());
        let mut out = Vec::new();
        if skip == 1 {
            out.extend(gamma.iter().map(|g| g[0]));
        }
        out.extend_from_slice(beta);
        for v in skip..self.ind_specific_vars.len() {
            out.extend(gamma.iter().map(|g| g[v]));
        }
        out
    }

    /// The coefficients in [`MlogitResult::coefficient_names`] order.
    pub fn coefficients(&self) -> Vec<f64> {
        self.in_mlogit_order(&self.beta, &self.gamma)
    }

    /// The standard errors in [`MlogitResult::coefficient_names`] order.
    pub fn std_errors(&self) -> Vec<f64> {
        self.in_mlogit_order(&self.beta_std_errors, &self.gamma_std_errors)
    }
}

/// Model and control of [`run_mlogit_with`] and [`run_conditional_logit_with`].
#[derive(Debug, Clone, PartialEq)]
pub struct MlogitConfig {
    /// J - 1 alternative-specific constants (mlogit's default). `false` fits
    /// `mlogit(choice ~ x | 0 + z)`, which for generic variables only is
    /// `survival::clogit`'s model.
    pub asc: bool,
    /// Reference alternative (mlogit's `reflevel`). `None`: the first alternative in factor
    /// order.
    pub reflevel: Option<String>,
    /// Maximum Newton iterations; at least 1.
    pub max_iter: usize,
    /// Converged when the Newton decrement `lambda2 = g' I^-1 g` (the predicted deviance
    /// decrease of the next step) satisfies `lambda2 / (|dev - lambda2| + 0.1) < tolerance`;
    /// that final full step is then taken. Positive and finite.
    pub tolerance: f64,
}

impl Default for MlogitConfig {
    fn default() -> Self {
        Self {
            asc: true,
            reflevel: None,
            max_iter: 100,
            tolerance: 1e-10,
        }
    }
}

impl ResultNotes for MlogitResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

impl fmt::Display for MlogitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "McFadden's Conditional Logit (mlogit)")?;
        writeln!(f, "======================================")?;
        writeln!(f, "N (choice situations) = {}", self.n_choice_situations)?;
        writeln!(
            f,
            "Alternatives = {} (reference: {})",
            self.n_alternatives, self.reference_alternative
        )?;
        writeln!(
            f,
            "Alternative-specific constants: {}",
            if self.asc { "yes" } else { "no" }
        )?;
        writeln!(f)?;

        if !self.alt_specific_vars.is_empty() {
            writeln!(
                f,
                "--- Alternative-Specific Variables (Generic Coefficients) ---"
            )?;
            writeln!(
                f,
                "{:<15} {:>10} {:>10} {:>10} {:>10}",
                "Variable", "Coef", "Std.Err", "z", "P>|z|"
            )?;
            writeln!(
                f,
                "{:-<15} {:-<10} {:-<10} {:-<10} {:-<10}",
                "", "", "", "", ""
            )?;
            for (i, var) in self.alt_specific_vars.iter().enumerate() {
                writeln!(
                    f,
                    "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    var,
                    self.beta[i],
                    self.beta_std_errors[i],
                    self.beta_z_stats[i],
                    self.beta_p_values[i]
                )?;
            }
            writeln!(f)?;
        }

        if !self.ind_specific_vars.is_empty() {
            let heading = if !self.has_constants() {
                "--- Individual-Specific Variables (Alternative-Specific Coefficients) ---"
            } else if self.ind_specific_vars.len() == 1 {
                "--- Alternative-Specific Constants ---"
            } else {
                "--- Alternative-Specific Constants and Individual-Specific Variables ---"
            };
            writeln!(f, "{heading}")?;
            for (alt_idx, alt) in self.alternatives.iter().enumerate() {
                if *alt == self.reference_alternative {
                    continue;
                }
                let coef_idx = if alt_idx == 0 {
                    0
                } else {
                    self.alternatives
                        .iter()
                        .take(alt_idx)
                        .filter(|a| **a != self.reference_alternative)
                        .count()
                };
                if coef_idx >= self.gamma.len() {
                    continue;
                }
                writeln!(f, "\n  {} vs {}:", alt, self.reference_alternative)?;
                writeln!(
                    f,
                    "  {:<13} {:>10} {:>10} {:>10} {:>10}",
                    "Variable", "Coef", "Std.Err", "z", "P>|z|"
                )?;
                for (var_idx, var) in self.ind_specific_vars.iter().enumerate() {
                    writeln!(
                        f,
                        "  {:<13} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                        var,
                        self.gamma[coef_idx][var_idx],
                        self.gamma_std_errors[coef_idx][var_idx],
                        self.gamma_z_stats[coef_idx][var_idx],
                        self.gamma_p_values[coef_idx][var_idx]
                    )?;
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        write!(f, "{}", notes_text(self))?;
        Ok(())
    }
}

/// Pre-computed data structure for fast mlogit computation.
struct MlogitData {
    /// For each choice situation: (start_row, n_alts, chosen_local_idx)
    choice_info: Vec<(usize, usize, usize)>,
    /// Pre-computed feature matrix
    features: Array2<f64>,
    /// Number of parameters
    n_params: usize,
}

impl MlogitData {
    /// `choice_situations[s]` lists the (row, alternative index) pairs of situation `s`, in
    /// data order; the situations are visited in index order.
    fn new(
        choice_situations: &[Vec<(usize, usize)>],
        choices: &[f64],
        x_alt_specific: &[&[f64]],
        z_ind_specific: &[&[f64]],
        n_alt_specific: usize,
        n_ind_specific: usize,
        n_alternatives: usize,
        ref_idx: usize,
    ) -> EconResult<Self> {
        let n_params = n_alt_specific + n_ind_specific * n_alternatives.saturating_sub(1);
        let total_rows: usize = choice_situations.iter().map(|v| v.len()).sum();

        let mut choice_info = Vec::with_capacity(choice_situations.len());
        let mut features =
            resource::try_zeros2(total_rows, n_params, "conditional logit feature matrix")?;

        let mut flat_idx = 0;
        for alts in choice_situations {
            let start = flat_idx;
            let n_alts = alts.len();
            let mut chosen_local = 0;

            for (local_idx, &(row_idx, alt_idx)) in alts.iter().enumerate() {
                // Fill feature row
                for (k, x_col) in x_alt_specific.iter().enumerate() {
                    features[[flat_idx, k]] = x_col[row_idx];
                }

                if alt_idx != ref_idx && n_ind_specific > 0 {
                    let gamma_offset = n_alt_specific
                        + (if alt_idx < ref_idx {
                            alt_idx
                        } else {
                            alt_idx - 1
                        }) * n_ind_specific;
                    for (k, z_col) in z_ind_specific.iter().enumerate() {
                        features[[flat_idx, gamma_offset + k]] = z_col[row_idx];
                    }
                }

                if choices[row_idx] > 0.5 {
                    chosen_local = local_idx;
                }

                flat_idx += 1;
            }

            choice_info.push((start, n_alts, chosen_local));
        }

        Ok(Self {
            choice_info,
            features,
            n_params,
        })
    }
}

fn compute_mlogit_loglik_fast(data: &MlogitData, params: &[f64]) -> f64 {
    let params_arr = ArrayView1::from(params);
    let mut ll = 0.0;

    for &(start, n_alts, chosen_local) in &data.choice_info {
        let mut max_v = f64::NEG_INFINITY;
        let mut chosen_v = 0.0;
        let mut sum_exp = 0.0;

        for i in 0..n_alts {
            let row = start + i;
            let v: f64 = data.features.row(row).dot(&params_arr);
            if v > max_v {
                max_v = v;
            }
            if i == chosen_local {
                chosen_v = v;
            }
        }

        for i in 0..n_alts {
            let row = start + i;
            let v: f64 = data.features.row(row).dot(&params_arr);
            sum_exp += (v - max_v).exp();
        }

        ll += chosen_v - max_v - sum_exp.ln();
    }

    ll
}

/// Log-likelihood, score and information at `params`. The score of a situation is
/// `x_chosen - xbar`, `xbar = sum_j p_j x_j`, and the information
/// `sum_j p_j (x_j - xbar)(x_j - xbar)'`, accumulated in that centred form (the
/// uncentred `sum p x x' - xbar xbar'` cancels catastrophically for large attributes).
fn compute_mlogit_derivatives_fast(
    data: &MlogitData,
    params: &[f64],
) -> EconResult<(f64, Array1<f64>, Array2<f64>)> {
    let m = data.n_params;
    let params_arr = ArrayView1::from(params);
    let mut ll = 0.0;
    let mut gradient = Array1::<f64>::zeros(m);
    let mut info = resource::try_zeros2(m, m, "conditional logit information matrix")?;

    let max_alts = data.choice_info.iter().map(|c| c.1).max().unwrap_or(0);
    let mut utilities = vec![0.0; max_alts];
    let mut probs = vec![0.0; max_alts];
    let mut xbar = vec![0.0; m];
    let mut dev = vec![0.0; m];

    for &(start, n_alts, chosen_local) in &data.choice_info {
        let mut max_v = f64::NEG_INFINITY;
        for i in 0..n_alts {
            let v = data.features.row(start + i).dot(&params_arr);
            utilities[i] = v;
            max_v = max_v.max(v);
        }
        let mut sum_exp = 0.0;
        for i in 0..n_alts {
            let e = (utilities[i] - max_v).exp();
            probs[i] = e;
            sum_exp += e;
        }
        for p in probs.iter_mut().take(n_alts) {
            *p /= sum_exp;
        }
        ll += utilities[chosen_local] - max_v - sum_exp.ln();

        xbar.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..n_alts {
            let row = data.features.row(start + i);
            for (xb, &x) in xbar.iter_mut().zip(row.iter()) {
                *xb += probs[i] * x;
            }
        }
        let chosen = data.features.row(start + chosen_local);
        for k in 0..m {
            gradient[k] += chosen[k] - xbar[k];
        }
        for i in 0..n_alts {
            let row = data.features.row(start + i);
            for k in 0..m {
                dev[k] = row[k] - xbar[k];
            }
            let p = probs[i];
            for k in 0..m {
                let pk = p * dev[k];
                if pk == 0.0 {
                    continue;
                }
                for l in k..m {
                    info[[k, l]] += pk * dev[l];
                }
            }
        }
    }
    for k in 0..m {
        for l in (k + 1)..m {
            info[[l, k]] = info[[k, l]];
        }
    }
    Ok((ll, gradient, info))
}

/// `s_k = 1 / sqrt(info_kk)` (1 for a zero diagonal entry) and `E = S info S`.
pub(super) fn equilibrate(info: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let scale: Array1<f64> = info
        .diag()
        .mapv(|d| if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 });
    let mut e = info.clone();
    for ((r, c), v) in e.indexed_iter_mut() {
        *v *= scale[r] * scale[c];
    }
    (scale, e)
}

/// The Newton step `d = info^-1 g` and the decrement `g' d`, solved on the equilibrated
/// information `E = S info S` by Cholesky; only when Cholesky refuses `E` (not positive
/// definite) or gives a non-finite step, by a symmetric generalized inverse, whose
/// warning is pushed into `warnings` once.
pub(super) fn equilibrated_newton_step(
    info: &Array2<f64>,
    grad: &Array1<f64>,
    what: &str,
    warnings: &mut Vec<String>,
) -> EconResult<(Array1<f64>, f64)> {
    let (scale, e) = equilibrate(info);
    let rhs: Array1<f64> = &scale * grad;
    let mut ds = None;
    if let Ok(chol) = ndarray_to_faer(&e.view()).llt(faer::Side::Lower) {
        let m = rhs.len();
        let b = faer::Mat::<f64>::from_fn(m, 1, |i, _| rhs[i]);
        let x = chol.solve(&b);
        let step: Array1<f64> = (0..m).map(|i| x[(i, 0)]).collect();
        if step.iter().all(|v| v.is_finite()) {
            ds = Some(step);
        }
    }
    let ds = match ds {
        Some(d) => d,
        None => {
            let inv = sym_inverse(e.view(), SingularPolicy::Generalized, what)?;
            if let Some(w) = inv.warning(what)
                && !warnings.contains(&w)
            {
                warnings.push(w);
            }
            inv.inverse.dot(&rhs)
        }
    };
    let d = &scale * &ds;
    let lambda2 = grad.dot(&d);
    Ok((d, lambda2))
}

/// Covariance from the information: `S E^-1 S` by [`sym_inverse`] with
/// [`SingularPolicy::Error`], or `None` with the warning when the information is
/// singular at the estimates.
pub(super) fn covariance_from_information(
    info: &Array2<f64>,
    what: &str,
    warnings: &mut Vec<String>,
) -> EconResult<Option<Array2<f64>>> {
    let (scale, e) = equilibrate(info);
    match sym_inverse(e.view(), SingularPolicy::Error, what) {
        Ok(inv) => {
            let mut v = inv.inverse;
            for ((r, c), x) in v.indexed_iter_mut() {
                *x *= scale[r] * scale[c];
            }
            Ok(Some(v))
        }
        Err(err @ EconError::SingularMatrix { .. }) => {
            warnings.push(format!(
                "the information matrix is singular at the estimates ({err}); standard errors \
                 are not available"
            ));
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

/// Validate a Newton control: `max_iter >= 1` and a positive, finite `tolerance`.
pub(super) fn check_newton_control(max_iter: usize, tolerance: f64) -> EconResult<()> {
    if max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1 (got 0)".to_string(),
        });
    }
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("tolerance must be a positive, finite number (got {tolerance})"),
        });
    }
    Ok(())
}

/// Run McFadden's conditional logit model with J - 1 alternative-specific constants.
///
/// R equivalent: `mlogit::mlogit(choice ~ x | z, reflevel = reference)`: generic
/// coefficients for `alt_specific_cols`, alternative-specific coefficients for
/// `ind_specific_cols`, and a constant for every alternative but the reference (the
/// first alternative in factor order when `reference` is `None`). Without constants,
/// see [`run_mlogit_with`] and [`MlogitConfig::asc`].
///
/// Every column comes from one request with `NaPolicy::Fail`: a missing id, alternative,
/// choice or attribute is refused with `NullValues` naming the column, as R's mlogit
/// refuses it. The choice indicator and the attributes may be Int64 or Boolean; a text
/// attribute is `NonNumericColumn`. Choice situations and alternatives follow R `factor()`
/// order (numeric ids numerically, text by byte), so the default reference alternative is
/// the first in that order.
pub fn run_mlogit(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    alt_specific_cols: &[&str],
    ind_specific_cols: &[&str],
    reference: Option<&str>,
) -> EconResult<MlogitResult> {
    let config = MlogitConfig {
        reflevel: reference.map(str::to_owned),
        ..MlogitConfig::default()
    };
    run_mlogit_with(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        alt_specific_cols,
        ind_specific_cols,
        &config,
    )
}

/// Convenience function for conditional logit with only alternative-specific variables
/// (and, by default, J - 1 alternative-specific constants).
///
/// R equivalent: `mlogit::mlogit(choice ~ x, reflevel = reference)`; for
/// `mlogit(choice ~ x | 0)` (= `survival::clogit`) use [`run_conditional_logit_with`] with
/// `asc = false`.
pub fn run_conditional_logit(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    reference: Option<&str>,
) -> EconResult<MlogitResult> {
    run_mlogit(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        &[],
        reference,
    )
}

/// [`run_mlogit`] with a model and control ([`MlogitConfig`]).
///
/// With `asc` the constants are an individual-specific variable named `(Intercept)`
/// (mlogit's `(Intercept):alt`). The model is fitted by Newton's method on the centred
/// information, solved on the equilibrated matrix; it stops when the Newton decrement on
/// `glm.fit`'s relative-deviance scale is below `tolerance`, then takes that last full
/// step. A line search that finds no increase stops the fit as not converged.
///
/// Refused: invalid controls and an unknown `reflevel` (`InvalidSpecification`), and an
/// attribute that is constant within every choice situation or a linear combination of
/// the others (`SingularMatrix` naming it; mlogit stops with "system is exactly
/// singular"). A singular information at the estimates (separation) gives `NaN`
/// standard errors and a warning.
pub fn run_mlogit_with(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    alt_specific_cols: &[&str],
    ind_specific_cols: &[&str],
    config: &MlogitConfig,
) -> EconResult<MlogitResult> {
    check_newton_control(config.max_iter, config.tolerance)?;
    let df = dataset.df();
    let cols = ColumnRequest::new(df)
        .factor(choice_id_col)
        .factor(alt_id_col)
        .numeric(choice_col)
        .numerics(alt_specific_cols)
        .numerics(ind_specific_cols)
        .na(NaPolicy::Fail)
        .extract()?;
    let n_rows = cols.n();
    let situation = cols.factor(choice_id_col);
    let alt = cols.factor(alt_id_col);
    let choices = cols.numeric(choice_col);
    let n_choice_situations = situation.n_levels();

    let alternatives: Vec<String> = alt.levels.clone();
    let n_alternatives = alternatives.len();

    if n_alternatives < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Conditional logit requires at least 2 alternatives".to_string(),
        });
    }

    let ref_alt = config
        .reflevel
        .clone()
        .unwrap_or_else(|| alternatives[0].clone());
    let Some(ref_idx) = alternatives.iter().position(|a| a == &ref_alt) else {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Reference alternative '{ref_alt}' not found in data; reflevel must be one of: {}",
                alternatives.join(", ")
            ),
        });
    };

    // Rows of each choice situation (in level order), in data order
    let mut choice_situations: Vec<Vec<(usize, usize)>> = resource::try_vec(
        n_choice_situations,
        Vec::new(),
        "conditional logit situations",
    )?;
    for (row_idx, (&s, &a)) in situation.codes.iter().zip(&alt.codes).enumerate() {
        choice_situations[s as usize].push((row_idx, a as usize));
    }

    let n_alt_specific = alt_specific_cols.len();
    let x_alt_specific: Vec<&[f64]> = alt_specific_cols.iter().map(|c| cols.numeric(c)).collect();

    // The constants are a virtual individual-specific column of ones, "(Intercept)".
    let ones = if config.asc {
        resource::try_vec(n_rows, 1.0, "conditional logit constants")?
    } else {
        Vec::new()
    };
    let mut ind_names: Vec<String> = Vec::new();
    let mut z_ind_specific: Vec<&[f64]> = Vec::new();
    if config.asc {
        ind_names.push("(Intercept)".to_string());
        z_ind_specific.push(&ones);
    }
    for c in ind_specific_cols {
        ind_names.push(c.to_string());
        z_ind_specific.push(cols.numeric(c));
    }
    let n_ind_specific = ind_names.len();

    let n_gamma = n_ind_specific * (n_alternatives - 1);
    let n_params = n_alt_specific + n_gamma;

    if n_params == 0 {
        return Err(EconError::InvalidSpecification {
            message: "At least one variable must be specified".to_string(),
        });
    }
    ensure_dense_inverse_fits(n_params, "conditional logit information matrix")?;

    let mlogit_data = MlogitData::new(
        &choice_situations,
        choices,
        &x_alt_specific,
        &z_ind_specific,
        n_alt_specific,
        n_ind_specific,
        n_alternatives,
        ref_idx,
    )?;

    // Parameter names in the internal layout (generic, then per non-reference
    // alternative its individual-specific variables), in mlogit's spelling.
    let non_ref: Vec<&String> = alternatives
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != ref_idx)
        .map(|(_, a)| a)
        .collect();
    let mut param_names: Vec<String> = alt_specific_cols.iter().map(|c| c.to_string()).collect();
    for a in &non_ref {
        for v in &ind_names {
            param_names.push(format!("{v}:{a}"));
        }
    }

    check_identified(&mlogit_data, &param_names)?;

    // Newton's method
    let mut params = vec![0.0; n_params];
    let mut warnings: Vec<String> = Vec::new();
    let what = "conditional logit information matrix";
    let (converged, iterations, final_change) =
        newton_maximise(&mlogit_data, &mut params, config, what, &mut warnings)?;
    let convergence = Convergence::new(
        converged,
        iterations,
        config.max_iter,
        "Newton decrement (relative deviance change)",
        final_change,
        config.tolerance,
    );
    if let Some(note) = convergence.note("conditional logit") {
        warnings.push(note);
    }

    let (log_likelihood, _, info) = compute_mlogit_derivatives_fast(&mlogit_data, &params)?;
    let log_likelihood_null = n_choice_situations as f64 * (-(n_alternatives as f64).ln());

    let std_errors: Vec<f64> = match covariance_from_information(&info, what, &mut warnings)? {
        Some(vcov) => (0..n_params)
            .map(|i| vcov[[i, i]].max(0.0).sqrt())
            .collect(),
        None => vec![f64::NAN; n_params],
    };

    // Separation: fitted probabilities numerically 0 or 1
    let (n_extreme, n_situations_extreme) = extreme_probabilities(&mlogit_data, &params);
    let separation = n_extreme > 0;
    if separation {
        warnings.push(format!(
            "Separation: fitted probabilities are numerically 0 or 1 for {n_extreme} of \
             {n_rows} alternatives in {n_situations_extreme} choice situations; the variables \
             (almost) perfectly predict the choices, so the maximum likelihood estimates do not \
             exist and the affected coefficients and standard errors are not reliable"
        ));
    }

    let z_of = |b: f64, se: f64| {
        if se > 0.0 && se.is_finite() {
            b / se
        } else {
            f64::NAN
        }
    };

    let beta: Vec<f64> = params[..n_alt_specific].to_vec();
    let beta_std_errors: Vec<f64> = std_errors[..n_alt_specific].to_vec();
    let beta_z_stats: Vec<f64> = beta
        .iter()
        .zip(&beta_std_errors)
        .map(|(&b, &se)| z_of(b, se))
        .collect();
    let beta_p_values: Vec<f64> = beta_z_stats
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    let mut gamma: Vec<Vec<f64>> = Vec::new();
    let mut gamma_std_errors: Vec<Vec<f64>> = Vec::new();
    let mut gamma_z_stats: Vec<Vec<f64>> = Vec::new();
    let mut gamma_p_values: Vec<Vec<f64>> = Vec::new();

    if n_ind_specific > 0 {
        for offset in 0..n_alternatives - 1 {
            let start = n_alt_specific + offset * n_ind_specific;
            let end = start + n_ind_specific;
            let g: Vec<f64> = params[start..end].to_vec();
            let g_se: Vec<f64> = std_errors[start..end].to_vec();
            let g_z: Vec<f64> = g.iter().zip(&g_se).map(|(&b, &se)| z_of(b, se)).collect();
            let g_p: Vec<f64> = g_z.iter().map(|&z| normal_p_value_two_sided(z)).collect();
            gamma.push(g);
            gamma_std_errors.push(g_se);
            gamma_z_stats.push(g_z);
            gamma_p_values.push(g_p);
        }
    }

    let mut choice_counts = vec![0usize; n_alternatives];
    for (&c, &a) in choices.iter().zip(&alt.codes).take(n_rows) {
        if c > 0.5 {
            choice_counts[a as usize] += 1;
        }
    }

    let pseudo_r_squared = 1.0 - log_likelihood / log_likelihood_null;
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n_choice_situations as f64).ln();

    Ok(MlogitResult {
        alt_specific_vars: alt_specific_cols.iter().map(|s| s.to_string()).collect(),
        ind_specific_vars: ind_names,
        alternatives: alternatives.clone(),
        reference_alternative: ref_alt,
        beta,
        beta_std_errors,
        beta_z_stats,
        beta_p_values,
        gamma,
        gamma_std_errors,
        gamma_z_stats,
        gamma_p_values,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic,
        bic,
        iterations,
        converged,
        n_choice_situations,
        n_alternatives,
        choice_counts,
        asc: config.asc,
        separation,
        convergence,
        warnings,
    })
}

/// Refuse a design whose parameters are not identified: the features centred within
/// each choice situation must have full column rank (pivoted QR at `LM_QR_TOL`). A
/// column whose centred norm is negligible against its raw norm (constant within every
/// situation, up to rounding of the means) counts as zero.
fn check_identified(data: &MlogitData, names: &[String]) -> EconResult<()> {
    let (rows, m) = data.features.dim();
    let mut centred = resource::try_zeros2(rows, m, "conditional logit identification check")?;
    for &(start, n_alts, _) in &data.choice_info {
        let block = data.features.slice(ndarray::s![start..start + n_alts, ..]);
        let mean = block.mean_axis(ndarray::Axis(0));
        for i in 0..n_alts {
            for k in 0..m {
                centred[[start + i, k]] = block[[i, k]] - mean.as_ref().map_or(0.0, |mu| mu[k]);
            }
        }
    }
    for k in 0..m {
        let raw = data.features.column(k).dot(&data.features.column(k)).sqrt();
        let cen = centred.column(k).dot(&centred.column(k)).sqrt();
        if cen <= LM_QR_TOL * raw {
            centred.column_mut(k).fill(0.0);
        }
    }
    let qr = Qr::decompose(centred.view(), LM_QR_TOL)?;
    let aliased = qr.aliased();
    if aliased.is_empty() {
        return Ok(());
    }
    let list: Vec<&str> = aliased.iter().map(|&j| names[j].as_str()).collect();
    Err(EconError::SingularMatrix {
        context: format!(
            "conditional logit: {} not identified (constant within every choice situation, or \
             a linear combination of the other variables; rank {} of {m})",
            list.join(", "),
            qr.rank()
        ),
        suggestion: "drop the listed variables (mlogit stops with 'system is exactly singular')"
            .to_string(),
    })
}

/// Newton's method on the conditional-logit log-likelihood from `params`; returns
/// (converged, iterations, last criterion value). Converged when the Newton decrement on
/// glm.fit's scale, `lambda2 / (|dev - lambda2| + 0.1)`, is below the tolerance; the last
/// full step is then taken when it does not lower the log-likelihood.
fn newton_maximise(
    data: &MlogitData,
    params: &mut [f64],
    config: &MlogitConfig,
    what: &str,
    warnings: &mut Vec<String>,
) -> EconResult<(bool, usize, f64)> {
    let mut final_change = f64::NAN;
    let mut iterations = 0;
    while iterations < config.max_iter {
        resource::checkpoint()?;
        let (ll, grad, info) = compute_mlogit_derivatives_fast(data, params)?;
        let (d, lambda2) = equilibrated_newton_step(&info, &grad, what, warnings)?;
        let dev = -2.0 * ll;
        final_change = deviance_change(dev - lambda2, dev);
        let trial =
            |t: f64| -> Vec<f64> { params.iter().zip(&d).map(|(p, s)| p + t * s).collect() };
        if final_change < config.tolerance {
            let full = trial(1.0);
            if compute_mlogit_loglik_fast(data, &full) >= ll - 1e-12 * ll.abs() {
                params.copy_from_slice(&full);
            }
            return Ok((true, iterations, final_change));
        }
        let mut t = 1.0;
        let mut accepted = None;
        for _ in 0..=MAX_HALVINGS {
            let cand = trial(t);
            if compute_mlogit_loglik_fast(data, &cand) > ll {
                accepted = Some(cand);
                break;
            }
            t *= 0.5;
        }
        let Some(cand) = accepted else {
            warnings.push(format!(
                "line search failed at iteration {}: no step along the Newton direction \
                 increases the log-likelihood; the estimates are not at the optimum",
                iterations + 1
            ));
            return Ok((false, iterations, final_change));
        };
        params.copy_from_slice(&cand);
        iterations += 1;
    }
    Ok((false, iterations, final_change))
}

/// (fitted probabilities within `10 eps` of 0 or 1, situations with at least one).
fn extreme_probabilities(data: &MlogitData, params: &[f64]) -> (usize, usize) {
    let params_arr = ArrayView1::from(params);
    let eps10 = 10.0 * f64::EPSILON;
    let (mut n_extreme, mut n_sit) = (0, 0);
    let mut v = Vec::new();
    for &(start, n_alts, _) in &data.choice_info {
        v.clear();
        v.extend((0..n_alts).map(|i| data.features.row(start + i).dot(&params_arr)));
        let max_v = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = v.iter().map(|u| (u - max_v).exp()).sum();
        let k = v
            .iter()
            .map(|u| (u - max_v).exp() / sum)
            .filter(|&p| p < eps10 || p > 1.0 - eps10)
            .count();
        n_extreme += k;
        n_sit += usize::from(k > 0);
    }
    (n_extreme, n_sit)
}

/// [`run_conditional_logit`] with a model and control ([`MlogitConfig`]).
pub fn run_conditional_logit_with(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    config: &MlogitConfig,
) -> EconResult<MlogitResult> {
    run_mlogit_with(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        &[],
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    /// Case data for tests that compare no R value.
    fn case_data(id: &str) -> (p2a_validation::Case, Dataset) {
        crate::test_support::load_case(id)
    }

    fn null_values(r: EconResult<MlogitResult>) -> (String, usize) {
        match r {
            Err(EconError::NullValues { column, count }) => (column, count),
            other => panic!("expected NullValues, got {other:?}"),
        }
    }

    /// A missing choice, attribute or id is refused, as R's mlogit refuses it; it was read
    /// as 0 (or grouped as one situation "").
    #[test]
    fn test_na_conditional_logit_missing_cells_refused() {
        let (_, ds) = case_data("disc_choice_na");
        let cl = |id: &str, choice: &str, x: &[&str]| {
            run_conditional_logit(&ds, id, "alt", choice, x, None)
        };
        assert_eq!(
            null_values(cl("id", "choice_na", &["x", "w"])),
            ("choice_na".to_string(), 2)
        );
        assert_eq!(
            null_values(cl("id", "choice", &["x_na", "w"])),
            ("x_na".to_string(), 1)
        );
        assert_eq!(
            null_values(cl("id_na", "choice", &["x", "w"])),
            ("id_na".to_string(), 1)
        );
        let ml = |id: &str, choice: &str, x: &[&str]| {
            run_mlogit(&ds, id, "alt", choice, x, &["w"], None)
        };
        assert_eq!(
            null_values(ml("id", "choice_na", &["x"])),
            ("choice_na".to_string(), 2)
        );
        assert_eq!(
            null_values(ml("id", "choice", &["x_na"])),
            ("x_na".to_string(), 1)
        );
        assert_eq!(
            null_values(ml("id_na", "choice", &["x"])),
            ("id_na".to_string(), 1)
        );
    }

    /// An Int64 attribute gives the fit of its Float64 copy.
    #[test]
    fn test_na_conditional_logit_int64_attribute() {
        let (_, ds) = case_data("disc_choice_na");
        let mut df = ds.df().clone();
        let f = df
            .column("x_int")
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .with_name("x_int_f".into());
        df.with_column(f).unwrap();
        let ds = Dataset::new(df);
        let a = run_conditional_logit(&ds, "id", "alt", "choice", &["x_int", "w"], None).unwrap();
        let b = run_conditional_logit(&ds, "id", "alt", "choice", &["x_int_f", "w"], None).unwrap();
        let tol = Tol::new(1e-12, 1e-14);
        for (x, y) in a
            .beta
            .iter()
            .zip(&b.beta)
            .chain(a.beta_std_errors.iter().zip(&b.beta_std_errors))
        {
            assert!(tol.allows(*x, *y), "{x} vs {y}");
        }
        assert!(tol.allows(a.log_likelihood, b.log_likelihood));
    }

    #[test]
    fn test_na_conditional_logit_text_attribute_refused() {
        let (_, ds) = case_data("disc_choice_na");
        match run_conditional_logit(&ds, "id", "alt", "choice", &["alt"], None) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "alt"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
    }

    /// The conditional logit on mixed_logit.csv against mlogit's fit, Newton-polished in R
    /// (mlogit itself stops with |gradient| ~ 1e-3, 1.5e-5 relative from the MLE).
    #[test]
    fn test_validate_conditional_logit_mixed_logit_data() {
        let (case, ds) = crate::test_support::load_case("disc_mixed_logit");
        // mlogit(choice ~ x + w | 0): no constants
        let mut cfg = MlogitConfig::default();
        cfg.asc = false;
        let r = run_conditional_logit_with(&ds, "id", "alt", "choice", &["x", "w"], &cfg).unwrap();
        Check::new(&case)
            .named(
                "clogit_coef_tight",
                &r.alt_specific_vars,
                &r.beta,
                Tol::iterative(1e-6),
            )
            .named(
                "clogit_se_tight",
                &r.alt_specific_vars,
                &r.beta_std_errors,
                Tol::iterative(1e-5),
            )
            .scalar("clogit_loglik_tight", r.log_likelihood, Tol::stat(300))
            .named(
                "clogit_coef",
                &r.alt_specific_vars,
                &r.beta,
                Tol::iterative(1e-4),
            )
            .finish();
        assert!(r.converged, "{r}");
        assert_eq!(r.n_choice_situations, 300);
    }

    fn create_mlogit_dataset() -> Dataset {
        let df = df! {
            "choice_id" => [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "alt_id" => ["car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train"],
            "choice" => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "cost" => [10.0, 3.0, 5.0, 8.0, 2.0, 4.0, 15.0, 4.0, 3.0,
                      5.0, 5.0, 8.0, 12.0, 2.0, 6.0],
            "time" => [20.0, 40.0, 30.0, 15.0, 35.0, 25.0, 25.0, 45.0, 20.0,
                      10.0, 30.0, 40.0, 20.0, 30.0, 25.0],
            "income" => [50.0, 50.0, 50.0, 30.0, 30.0, 30.0, 70.0, 70.0, 70.0,
                        60.0, 60.0, 60.0, 25.0, 25.0, 25.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_mlogit_basic() {
        let dataset = create_mlogit_dataset();
        let result = run_mlogit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[],
            None,
        )
        .unwrap();

        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);
        assert_eq!(result.beta.len(), 2);
        assert!(result.beta[0] < 0.0); // Cost should be negative
    }

    #[test]
    fn test_conditional_logit() {
        let dataset = create_mlogit_dataset();
        let result =
            run_conditional_logit(&dataset, "choice_id", "alt_id", "choice", &["cost"], None)
                .unwrap();

        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.beta.len(), 1);
        assert!(result.asc);
        assert_eq!(result.ind_specific_vars, ["(Intercept)"]);
        assert_eq!(result.gamma.len(), 2);
    }

    /// `to = from * scale` as a Float64 column (derived data; the case's dataset stays the
    /// source).
    fn with_scaled(ds: &Dataset, from: &str, to: &str, scale: f64) -> Dataset {
        crate::regression::ols::na_fixtures::with_derived(ds, to, from, |v| v * scale)
    }

    fn renamed(names: &[String], from: &str, to: &str) -> Vec<String> {
        names.iter().map(|n| n.replace(from, to)).collect()
    }

    /// mlogit(chosen ~ price + catch | income) on Fishing: J - 1 constants by default.
    #[test]
    fn test_validate_mlogit_fishing_asc() {
        let (case, ds) = crate::test_support::load_case("disc_mlogit_fishing");
        let r = run_mlogit(
            &ds,
            "id",
            "alt",
            "chosen",
            &["price", "catch"],
            &["income"],
            None,
        )
        .unwrap();
        let (names, n) = (r.coefficient_names(), r.n_choice_situations);
        let counts: Vec<f64> = r.choice_counts.iter().map(|&c| c as f64).collect();
        Check::new(&case)
            .scalar("n_choices", n as f64, Tol::exact())
            .named("choice_counts", &r.alternatives, &counts, Tol::exact())
            .named(
                "asc_coef_tight",
                &names,
                &r.coefficients(),
                Tol::iterative(1e-9),
            )
            .named(
                "asc_se_tight",
                &names,
                &r.std_errors(),
                Tol::iterative(1e-8),
            )
            .scalar("asc_loglik", r.log_likelihood, Tol::stat(n))
            .scalar("asc_aic", r.aic, Tol::stat(n))
            // mlogit's own optimum
            .named("asc_coef", &names, &r.coefficients(), Tol::iterative(1e-6))
            .named("asc_se", &names, &r.std_errors(), Tol::iterative(1e-6))
            .finish();
        assert!(r.asc && r.converged && !r.separation && r.warnings.is_empty());
        assert_eq!(r.reference_alternative, "beach");
        assert_eq!(r.ind_specific_vars, ["(Intercept)", "income"]);
        let text = r.to_string();
        assert!(
            text.contains("Alternative-specific constants: yes"),
            "{text}"
        );
        assert!(text.contains("Converged: yes ("), "{text}");
    }

    /// `asc = false` is mlogit(chosen ~ price + catch | 0 + income), a different model.
    #[test]
    fn test_validate_mlogit_fishing_noasc() {
        let (case, ds) = crate::test_support::load_case("disc_mlogit_fishing");
        let mut cfg = MlogitConfig::default();
        cfg.asc = false;
        let r = run_mlogit_with(
            &ds,
            "id",
            "alt",
            "chosen",
            &["price", "catch"],
            &["income"],
            &cfg,
        )
        .unwrap();
        let names = r.coefficient_names();
        Check::new(&case)
            .named(
                "noasc_coef_tight",
                &names,
                &r.coefficients(),
                Tol::iterative(1e-9),
            )
            .named(
                "noasc_se_tight",
                &names,
                &r.std_errors(),
                Tol::iterative(1e-8),
            )
            .scalar("noasc_loglik", r.log_likelihood, Tol::stat(1182))
            .named(
                "noasc_coef",
                &names,
                &r.coefficients(),
                Tol::iterative(1e-6),
            )
            .finish();
        assert!(!r.asc && r.converged);
        let a = run_mlogit(
            &ds,
            "id",
            "alt",
            "chosen",
            &["price", "catch"],
            &["income"],
            None,
        )
        .unwrap();
        assert!(a.log_likelihood - r.log_likelihood > 40.0);
    }

    /// Generic variables only, with constants (mlogit's default) and without (`| 0`).
    #[test]
    fn test_validate_conditional_logit_fishing_generic() {
        let (case, ds) = crate::test_support::load_case("disc_mlogit_fishing");
        let r =
            run_conditional_logit(&ds, "id", "alt", "chosen", &["price", "catch"], None).unwrap();
        let names = r.coefficient_names();
        let mut cfg = MlogitConfig::default();
        cfg.asc = false;
        let z = run_conditional_logit_with(&ds, "id", "alt", "chosen", &["price", "catch"], &cfg)
            .unwrap();
        let zn = z.coefficient_names();
        Check::new(&case)
            .named(
                "generic_asc_coef_tight",
                &names,
                &r.coefficients(),
                Tol::iterative(1e-9),
            )
            .named(
                "generic_asc_se_tight",
                &names,
                &r.std_errors(),
                Tol::iterative(1e-8),
            )
            .scalar("generic_asc_loglik", r.log_likelihood, Tol::stat(1182))
            .named(
                "generic_asc_coef",
                &names,
                &r.coefficients(),
                Tol::iterative(1e-6),
            )
            .named(
                "generic_noasc_coef_tight",
                &zn,
                &z.coefficients(),
                Tol::iterative(1e-9),
            )
            .named(
                "generic_noasc_se_tight",
                &zn,
                &z.std_errors(),
                Tol::iterative(1e-8),
            )
            .scalar("generic_noasc_loglik", z.log_likelihood, Tol::stat(1182))
            .finish();
        assert_eq!(r.ind_specific_vars, ["(Intercept)"]);
        assert!(r.converged && z.converged);
    }

    /// reflevel = "pier": the same model in another parameterisation.
    #[test]
    fn test_validate_mlogit_fishing_reflevel() {
        let (case, ds) = crate::test_support::load_case("disc_mlogit_fishing");
        let r = run_mlogit(
            &ds,
            "id",
            "alt",
            "chosen",
            &["price", "catch"],
            &["income"],
            Some("pier"),
        )
        .unwrap();
        let names = r.coefficient_names();
        assert_eq!(r.reference_alternative, "pier");
        Check::new(&case)
            .named(
                "pier_coef_tight",
                &names,
                &r.coefficients(),
                Tol::iterative(1e-9),
            )
            .named(
                "pier_se_tight",
                &names,
                &r.std_errors(),
                Tol::iterative(1e-8),
            )
            .scalar("pier_loglik", r.log_likelihood, Tol::stat(1182))
            .finish();
    }

    /// Newton's method is affine invariant, so rescaling income by 1e-6 or 1e6 changes
    /// the iteration count by at most one (rounding at the stopping threshold) and the
    /// coefficients only by the scale.
    #[test]
    fn test_validate_mlogit_convergence_is_scale_free() {
        let (case, ds) = crate::test_support::load_case("disc_mlogit_fishing");
        let base = run_mlogit(
            &ds,
            "id",
            "alt",
            "chosen",
            &["price", "catch"],
            &["income"],
            None,
        )
        .unwrap();
        for scale in [1e-6, 1e6] {
            let s = with_scaled(&ds, "income", "income_s", scale);
            let r = run_mlogit(
                &s,
                "id",
                "alt",
                "chosen",
                &["price", "catch"],
                &["income_s"],
                None,
            )
            .unwrap();
            assert!(r.converged, "scale {scale}: {:?}", r.warnings);
            assert!(
                r.iterations.abs_diff(base.iterations) <= 1,
                "scale {scale}: {} vs {}",
                r.iterations,
                base.iterations
            );
            let names = renamed(&r.coefficient_names(), "income_s:", "income:");
            let back: Vec<f64> = names
                .iter()
                .zip(r.coefficients())
                .map(|(n, c)| {
                    if n.starts_with("income:") {
                        c * scale
                    } else {
                        c
                    }
                })
                .collect();
            Check::new(&case)
                .named("asc_coef_tight", &names, &back, Tol::iterative(1e-9))
                .finish();
        }
    }

    /// An attribute that is a multiple of another, or constant within every choice
    /// situation, is not identified: an error naming it (mlogit stops too).
    #[test]
    fn test_validate_mlogit_aliased_attribute_is_an_error() {
        let (case, ds) = crate::test_support::load_case("disc_mlogit_fishing");
        assert_eq!(case.scalar("aliased_refused"), 1.0);
        let s = with_scaled(&ds, "price", "price2", 2.0);
        match run_conditional_logit(
            &s,
            "id",
            "alt",
            "chosen",
            &["price", "price2", "catch"],
            None,
        ) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(context.contains("price2"), "{context}")
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
        match run_conditional_logit(
            &ds,
            "id",
            "alt",
            "chosen",
            &["price", "catch", "income"],
            None,
        ) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(context.contains("income"), "{context}")
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
    }

    /// 5 situations: cost | income with constants predicts every choice.
    #[test]
    fn test_mlogit_separation_warns() {
        let r = run_mlogit(
            &create_mlogit_dataset(),
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            &["income"],
            None,
        )
        .unwrap();
        assert!(r.separation);
        assert!(
            r.warnings.iter().any(|w| w.starts_with("Separation:")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_mlogit_config_rejects_invalid_controls() {
        let ds = create_mlogit_dataset();
        for (max_iter, tolerance, needle) in [
            (0, 1e-10, "max_iter"),
            (100, 0.0, "tolerance"),
            (100, f64::INFINITY, "tolerance"),
        ] {
            let mut cfg = MlogitConfig::default();
            cfg.max_iter = max_iter;
            cfg.tolerance = tolerance;
            match run_conditional_logit_with(&ds, "choice_id", "alt_id", "choice", &["cost"], &cfg)
            {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains(needle), "{message}")
                }
                other => panic!("expected InvalidSpecification naming {needle}, got {other:?}"),
            }
        }
    }

    /// A reference alternative that is not in the data is an error naming it.
    #[test]
    fn test_mlogit_unknown_reference_is_an_error() {
        let ds = create_mlogit_dataset();
        let mut cfg = MlogitConfig::default();
        cfg.reflevel = Some("bike".into());
        assert!(matches!(
            run_conditional_logit_with(&ds, "choice_id", "alt_id", "choice", &["cost"], &cfg),
            Err(EconError::InvalidSpecification { message }) if message.contains("'bike'")
        ));
        assert!(matches!(
            run_mlogit(&ds, "choice_id", "alt_id", "choice", &["cost"], &[], Some("bike")),
            Err(EconError::InvalidSpecification { message }) if message.contains("'bike'")
        ));
    }
}
