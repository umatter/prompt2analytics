//! The Granger causality F test ([`granger_test`], [`run_granger_test`], [`granger_test_bidirectional`]).

use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{LM_QR_TOL, Qr};
use crate::regression::ols::normal_equations_inverse;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::f_test_p_value;

// =============================================================================
// Granger Causality Test
// =============================================================================

/// Result of a Granger causality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerResult {
    /// F statistic for the test
    pub f_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom (numerator)
    pub df1: usize,
    /// Degrees of freedom (denominator)
    pub df2: usize,
    /// Number of lags used
    pub lags: usize,
    /// Number of observations used in estimation
    pub n_obs: usize,
    /// Name of the dependent (caused) variable
    pub dependent: String,
    /// Name of the potential causing variable
    pub cause: String,
    /// RSS of unrestricted model
    pub rss_unrestricted: f64,
    /// RSS of restricted model
    pub rss_restricted: f64,
    /// Whether significant at 0.05 level
    pub significant_at_05: bool,
    /// Interpretation of the result
    pub interpretation: String,
    /// Warnings (a reduced default lag order)
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for GrangerResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for GrangerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Granger Causality Test")?;
        writeln!(f, "======================")?;
        writeln!(
            f,
            "H0: {} does not Granger-cause {}",
            self.cause, self.dependent
        )?;
        writeln!(f)?;
        writeln!(f, "F Statistic: {:.4}", self.f_statistic)?;
        writeln!(f, "df: ({}, {})", self.df1, self.df2)?;
        writeln!(f, "p-value: {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "Lags: {}", self.lags)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f)?;
        writeln!(f, "{}", self.interpretation)?;
        write!(f, "{}", notes_text(self))?;
        Ok(())
    }
}

/// Perform Granger causality test.
///
/// Tests whether the lagged values of variable `cause` help predict variable `dependent`,
/// after controlling for lagged values of `dependent` itself.
///
/// # Mathematical Background
///
/// The test compares two regression models:
///
/// **Restricted model** (H0: no Granger causality):
/// yₜ = α + Σᵢ βᵢ yₜ₋ᵢ + εₜ
///
/// **Unrestricted model**:
/// yₜ = α + Σᵢ βᵢ yₜ₋ᵢ + Σⱼ γⱼ xₜ₋ⱼ + εₜ
///
/// The F-statistic is:
/// F = [(RSS_R - RSS_U) / p] / [RSS_U / (n - 3p - 1)]
///
/// Under H0, F ~ F(p, n - 3p - 1), n = number of rows (each regression uses the n - p rows
/// t = p + 1..n). Both regressions are solved as OLS solves them: the normal equations when
/// the designs are of full rank and well conditioned, otherwise one pivoted QR of the
/// unrestricted design (`lm.fit`'s, which decides rank and names aliased lags).
///
/// # Requirements
///
/// `n >= 3p + 2`, so that df2 >= 1 (`InsufficientData` otherwise; lmtest::grangertest stops
/// with "there are aliased coefficients in the model"), and lags of full rank (identical or
/// proportional series are `SingularMatrix`).
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `dependent` - Name of the dependent variable column (the "caused" variable)
/// * `cause` - Name of the potential causing variable column
/// * `lags` - Number of lags to include (at least 1)
///
/// # Returns
///
/// `GrangerResult` containing F-statistic, p-value, and interpretation.
///
/// # References
///
/// - Granger, C.W.J. (1969). Investigating Causal Relations by Econometric Models and
///   Cross-spectral Methods. *Econometrica*, 37(3), 424-438.
///   <https://doi.org/10.2307/1912791>
///
/// R equivalent: `lmtest::grangertest()`
///
/// Integer and Boolean columns are accepted; a missing value (null or NaN) in either
/// series is `EconError::NullValues` naming the column. R's grangertest drops rows instead,
/// but its formula and default interfaces disagree on which rows; fill the gaps first.
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::granger_test;
///
/// // Test whether x Granger-causes y
/// let result = granger_test(&dataset, "y", "x", 2)?;
/// println!("{}", result);
/// ```
pub fn granger_test(
    dataset: &Dataset,
    dependent: &str,
    cause: &str,
    lags: usize,
) -> EconResult<GrangerResult> {
    if lags == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Number of lags must be at least 1".to_string(),
        });
    }

    // Extract the two time series: Int64 accepted, a missing value refused by name
    let cols = ColumnRequest::new(dataset.df())
        .numerics(&[dependent, cause])
        .na(NaPolicy::Fail)
        .extract()?;
    let y_vec = cols.numeric(dependent).to_vec();
    let x_vec = cols.numeric(cause).to_vec();

    let n_total = y_vec.len();
    if n_total != x_vec.len() {
        return Err(EconError::InvalidSpecification {
            message: "Variables must have the same length".to_string(),
        });
    }
    // Both regressions use the n - p rows t = p + 1..n; the unrestricted one has 2p + 1
    // coefficients, so df2 = (n - p) - (2p + 1) = n - 3p - 1 must be at least 1 (ECON-53).
    let required = lags.checked_mul(3).and_then(|v| v.checked_add(2));
    let df2 = match required {
        Some(req) if n_total >= req => n_total - 3 * lags - 1,
        _ => {
            return Err(EconError::InsufficientData {
                required: required.unwrap_or(usize::MAX),
                provided: n_total,
                context: format!(
                    "Granger causality test with {lags} lags: the unrestricted regression has \
                     2p + 1 = {} coefficients on n - p = {} rows, and df2 = n - 3p - 1 must be \
                     at least 1 (lmtest::grangertest stops with \"there are aliased coefficients \
                     in the model\")",
                    lags.saturating_mul(2).saturating_add(1),
                    n_total.saturating_sub(lags)
                ),
            });
        }
    };

    let n = n_total - lags;

    // Build dependent variable vector (y values from lags onwards)
    let y = Array1::from_vec(y_vec[lags..].to_vec());

    // Build unrestricted design matrix: intercept + lagged y + lagged x
    // X_U = [1, y_{t-1}, ..., y_{t-p}, x_{t-1}, ..., x_{t-p}]
    let mut x_unrestricted = resource::try_zeros2(n, 2 * lags + 1, "unrestricted Granger design")?;
    for i in 0..n {
        x_unrestricted[[i, 0]] = 1.0; // intercept
        for lag in 1..=lags {
            x_unrestricted[[i, lag]] = y_vec[lags + i - lag];
            x_unrestricted[[i, lags + lag]] = x_vec[lags + i - lag];
        }
    }

    // OLS's solve decision (regression::ols): the normal equations when both designs are of
    // full rank and well conditioned, otherwise one pivoted QR, which decides rank and names
    // the aliased lags as lm.fit does.
    let (rss_u, rss_gain) = match granger_normal_rss(&x_unrestricted, &y, lags)? {
        Some(v) => v,
        None => granger_qr_rss(&x_unrestricted, &y, lags, dependent, cause)?,
    };
    let rss_r = rss_u + rss_gain;

    // F without an absolute exact-fit threshold: it is scale-free. RSS_U = 0 gives F = inf
    // (p = 0) when RSS_R > RSS_U, and NaN when both are 0, as in R.
    let df1 = lags;
    let f_statistic = (rss_gain / df1 as f64) / (rss_u / df2 as f64);

    let p_value = f_test_p_value(f_statistic, df1 as f64, df2 as f64);
    let significant = p_value < 0.05;

    let interpretation = if significant {
        format!(
            "Reject H0 at 5% level: {} Granger-causes {} (p = {:.4}). \
             Lagged values of {} significantly improve predictions of {}.",
            cause, dependent, p_value, cause, dependent
        )
    } else {
        format!(
            "Cannot reject H0 at 5% level: {} does not Granger-cause {} (p = {:.4}). \
             No evidence that lagged values of {} improve predictions of {}.",
            cause, dependent, p_value, cause, dependent
        )
    };

    Ok(GrangerResult {
        f_statistic,
        p_value,
        df1,
        df2,
        lags,
        n_obs: n,
        dependent: dependent.to_string(),
        cause: cause.to_string(),
        rss_unrestricted: rss_u,
        rss_restricted: rss_r,
        significant_at_05: significant,
        interpretation,
        warnings: Vec::new(),
    })
}

/// `(RSS_U, RSS_R - RSS_U)` by the normal equations of the unrestricted design and of the
/// restricted one (its first p + 1 columns), or `None` when either is rank-deficient or ill
/// conditioned for them (the QR then decides). The gain is the sum of squares of the
/// difference of the two fitted vectors, `||X_U b_U - X_R b_R||^2`, which equals
/// RSS_R - RSS_U for nested least-squares fits and is never negative.
fn granger_normal_rss(
    x_unrestricted: &Array2<f64>,
    y: &Array1<f64>,
    lags: usize,
) -> EconResult<Option<(f64, f64)>> {
    let Some(unrestricted) = normal_equations_inverse(x_unrestricted.view())? else {
        return Ok(None);
    };
    let x_restricted = x_unrestricted.slice(s![.., ..=lags]).to_owned();
    let Some(restricted) = normal_equations_inverse(x_restricted.view())? else {
        return Ok(None);
    };
    let fit_u = x_unrestricted.dot(&unrestricted.coefficients(x_unrestricted.view(), y));
    let fit_r = x_restricted.dot(&restricted.coefficients(x_restricted.view(), y));
    let rss_u = y
        .iter()
        .zip(&fit_u)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>();
    let rss_gain = fit_u
        .iter()
        .zip(&fit_r)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>();
    Ok(Some((rss_u, rss_gain)))
}

/// `(RSS_U, RSS_R - RSS_U)` from one pivoted QR of the unrestricted design, or
/// `SingularMatrix` naming the aliased lags.
fn granger_qr_rss(
    x_unrestricted: &Array2<f64>,
    y: &Array1<f64>,
    lags: usize,
    dependent: &str,
    cause: &str,
) -> EconResult<(f64, f64)> {
    // The restricted design [1, y lags] is the first p + 1 columns of the unrestricted one.
    // At full rank dqrdc2 moves no column, so the first p + 1 reflectors of the unrestricted
    // QR are the QR of the restricted design, and one decomposition gives both residual sums
    // of squares from the effects Q'y: RSS_U = sum over i >= 2p + 1, RSS_R = sum over
    // i >= p + 1 (lm.fit on each design gives the same).
    let m = x_unrestricted.ncols();
    let n = x_unrestricted.nrows();
    let qr = Qr::decompose(x_unrestricted.view(), LM_QR_TOL)?;
    if qr.rank() < m {
        let aliased = qr.aliased();
        let model = if aliased.iter().any(|&j| j <= lags) {
            "restricted"
        } else {
            "unrestricted"
        };
        let labels: Vec<String> = aliased
            .iter()
            .map(|&j| match j {
                0 => "(Intercept)".to_string(),
                j if j <= lags => format!("{dependent}_{j}"),
                j => format!("{cause}_{}", j - lags),
            })
            .collect();
        return Err(EconError::SingularMatrix {
            context: format!(
                "{model} Granger regression has aliased coefficients ({}): the lags of \
                 {dependent} and {cause} are collinear",
                labels.join(", ")
            ),
            suggestion: "lmtest::grangertest stops here too; are the two series identical or \
                         proportional?"
                .to_string(),
        });
    }
    // RSS_R is read off the first p + 1 effects, which needs the identity pivot; checked in
    // release builds too (m comparisons) so a change to Qr's pivoting cannot go unnoticed.
    if !qr.pivot().iter().enumerate().all(|(i, &j)| i == j) {
        return Err(EconError::Computation(
            "Granger QR pivoted a full-rank design; restricted RSS undefined".to_string(),
        ));
    }
    let effects = qr.qty(y.view())?;
    let sum_sq = |from: usize, to: usize| {
        effects
            .slice(s![from..to])
            .iter()
            .map(|e| e * e)
            .sum::<f64>()
    };
    let rss_u = sum_sq(m, n);
    // RSS_R - RSS_U as a sum of squares: never negative
    let rss_gain = sum_sq(lags + 1, m);
    Ok((rss_u, rss_gain))
}

/// Granger causality test with the default lag order.
///
/// The order is Schwert's `floor(12 (n/100)^(1/4))` (Schwert, 1989), capped at
/// `(n - 2) / 3`, the largest order with at least one residual degree of freedom
/// (n - 3p - 1 >= 1). When the cap applies the result carries a warning. Fewer than 5 rows
/// is `InsufficientData`. lmtest::grangertest's own default order is 1.
pub fn run_granger_test(
    dataset: &Dataset,
    dependent: &str,
    cause: &str,
) -> EconResult<GrangerResult> {
    // Columns first, so a missing or text column is named as such on a short dataset too
    let n = ColumnRequest::new(dataset.df())
        .numerics(&[dependent, cause])
        .na(NaPolicy::Fail)
        .extract()?
        .rows()
        .len();
    let schwert = ((12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize).max(1);
    let cap = n.saturating_sub(2) / 3; // largest p with n - 3p - 1 >= 1
    if cap == 0 {
        return Err(EconError::InsufficientData {
            required: 5,
            provided: n,
            context: "Granger causality test: one lag needs n - 3p - 1 >= 1, i.e. at least 5 \
                      observations"
                .to_string(),
        });
    }
    let lags = schwert.min(cap);
    let mut result = granger_test(dataset, dependent, cause, lags)?;
    if lags < schwert {
        result.warnings.push(format!(
            "default lag order {schwert} (Schwert's rule floor(12 (n/100)^(1/4)), n = {n}) \
             reduced to {lags}, the largest order with at least one residual degree of freedom \
             (n - 3p - 1 >= 1); pass lags to choose the order (lmtest::grangertest's default is 1)"
        ));
    }
    Ok(result)
}

/// Test bidirectional Granger causality between two variables.
///
/// Runs the test in both directions and returns both results.
pub fn granger_test_bidirectional(
    dataset: &Dataset,
    var1: &str,
    var2: &str,
    lags: usize,
) -> EconResult<(GrangerResult, GrangerResult)> {
    let result1 = granger_test(dataset, var1, var2, lags)?; // var2 -> var1
    let result2 = granger_test(dataset, var2, var1, lags)?; // var1 -> var2
    Ok((result1, result2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::timeseries::common::test_fixtures::*;
    use crate::econometrics::timeseries::run_var;
    use polars::prelude::*;

    /// R: lmtest::grangertest(x = m, y = ai, order = 2 | 3) on the Int64 series ai.
    #[test]
    fn test_validate_granger_int_series() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("ts_granger_int");
        assert_eq!(ds.df().column("ai").unwrap().dtype(), &DataType::Int64);
        let n = 300;
        let mut check = Check::new(&case);
        for lags in [2usize, 3] {
            let r = granger_test(&ds, "ai", "m", lags).unwrap();
            let k = format!("lag{lags}");
            check
                .scalar(&format!("{k}_f"), r.f_statistic, Tol::stat(n))
                .scalar(&format!("{k}_p"), r.p_value, Tol::p_value(n))
                .scalar(&format!("{k}_df1"), r.df1 as f64, Tol::exact())
                .scalar(&format!("{k}_df2"), r.df2 as f64, Tol::exact())
                .scalar(&format!("{k}_n_used"), r.n_obs as f64, Tol::exact());
        }
        check.finish();
    }

    /// The dataset of case ts_granger_int (Int64 `ai`, `a_na`/`ai_na` with two nulls each,
    /// text `lab`), for tests that compare no case value.
    fn granger_series() -> Dataset {
        crate::test_support::load_case("ts_granger_int").1
    }

    /// X-2: Granger and the VAR family refuse missing values and text by name; integer
    /// columns run.
    #[test]
    fn test_na_granger_var_missing_and_text() {
        let ds = granger_series();
        let is_null = |e: &EconError, column: &str| matches!(e, EconError::NullValues { column: c, count: 2 } if c == column);
        let is_text =
            |e: &EconError| matches!(e, EconError::NonNumericColumn { column } if column == "lab");
        let e = granger_test(&ds, "a_na", "m", 2).unwrap_err();
        assert!(is_null(&e, "a_na"), "{e:?}");
        let e = granger_test(&ds, "ai", "ai_na", 2).unwrap_err();
        assert!(is_null(&e, "ai_na"), "{e:?}");
        let e = granger_test(&ds, "lab", "m", 2).unwrap_err();
        assert!(is_text(&e), "{e:?}");
        let e = run_var(&ds, &["a", "a_na"], 1).unwrap_err();
        assert!(is_null(&e, "a_na"), "{e:?}");
        let e = run_var(&ds, &["a", "lab"], 1).unwrap_err();
        assert!(is_text(&e), "{e:?}");
        run_var(&ds, &["ai", "m"], 1).unwrap();
    }

    /// The dataset of case granger_short (20 rows), for tests that compare no case value.
    fn short_series() -> Dataset {
        crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/granger_short.csv",
        )
    }

    /// R: lmtest::grangertest(y ~ x, order = p), p = 1..6, on granger_short (n = 20;
    /// df2 = n - 3p - 1). run_granger_test: Schwert's floor(12 (n/100)^(1/4)) = 8 is capped
    /// at (n - 2) / 3 = 6 so that df2 >= 1 (p2a's rule, written out in the case; R's own
    /// default order is 1).
    #[test]
    fn test_validate_granger_short_lags() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("granger_short");
        let n = 20;
        let mut check = Check::new(&case);
        for p in 1..=6usize {
            let r = granger_test(&ds, "y", "x", p).unwrap();
            let k = format!("lag{p}");
            check
                .scalar(&format!("{k}_F"), r.f_statistic, Tol::stat(n))
                .scalar(&format!("{k}_p"), r.p_value, Tol::p_value(n))
                .scalar(&format!("{k}_df1"), r.df1 as f64, Tol::exact())
                .scalar(&format!("{k}_df2"), r.df2 as f64, Tol::exact())
                .scalar(&format!("{k}_n_used"), r.n_obs as f64, Tol::exact());
            assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        }
        let d = run_granger_test(&ds, "y", "x").unwrap();
        check
            .scalar("default_lag", d.lags as f64, Tol::exact())
            .scalar("lag6_F", d.f_statistic, Tol::stat(n))
            .scalar("lag6_p", d.p_value, Tol::p_value(n))
            .finish();
        assert_eq!(case.scalar("schwert_lag"), 8.0);
        assert_eq!(d.warnings.len(), 1, "{:?}", d.warnings);
        assert!(
            d.warnings[0].starts_with("default lag order 8")
                && d.warnings[0].contains("reduced to 6"),
            "{}",
            d.warnings[0]
        );
        assert!(
            d.to_string().contains("Warning: default lag order 8"),
            "{d}"
        );
    }

    /// n < 3p + 2 leaves the unrestricted regression (2p + 1 coefficients on n - p rows)
    /// without residual df; R: grangertest(order = 7) on these 20 rows stops with "there are
    /// aliased coefficients in the model".
    #[test]
    fn test_granger_rejects_lags_without_residual_df() {
        let ds = short_series();
        for (p, need) in [(7usize, 23usize), (8, 26)] {
            match granger_test(&ds, "y", "x", p) {
                Err(EconError::InsufficientData {
                    required,
                    provided,
                    context,
                }) => {
                    assert_eq!((required, provided), (need, 20));
                    assert!(context.contains("n - 3p - 1"), "{context}");
                }
                other => panic!("lags = {p}: expected InsufficientData, got {other:?}"),
            }
        }
        let tiny = Dataset::new(
            df! { "y" => [0.3, -1.2, 0.8, 0.1], "x" => [1.0, 0.4, -0.7, 0.9] }.unwrap(),
        );
        assert!(matches!(
            run_granger_test(&tiny, "y", "x"),
            Err(EconError::InsufficientData {
                required: 5,
                provided: 4,
                ..
            })
        ));
        // a misspelled column is named, not reported as too few rows
        assert!(matches!(
            run_granger_test(&tiny, "y", "z"),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "z"
        ));
    }

    /// The exact-fit shortcut compared RSS with an absolute 1e-15: the same series in units
    /// 1e-10 smaller reported F = inf, p = 0. F is invariant to the scale of the data.
    #[test]
    fn test_granger_is_scale_invariant() {
        use crate::data::columns::{NaPolicy, numeric_column};
        let ds = short_series();
        let scaled = |name: &str| -> Vec<f64> {
            numeric_column(ds.df(), name, NaPolicy::Fail)
                .unwrap()
                .0
                .iter()
                .map(|v| v * 1e-10)
                .collect()
        };
        let small = Dataset::new(df! { "x" => scaled("x"), "y" => scaled("y") }.unwrap());
        for p in [1usize, 3] {
            let a = granger_test(&ds, "y", "x", p).unwrap();
            let b = granger_test(&small, "y", "x", p).unwrap();
            assert!(
                (b.f_statistic / a.f_statistic - 1.0).abs() < 1e-8,
                "lags {p}: {} vs {}",
                b.f_statistic,
                a.f_statistic
            );
        }
    }

    /// x = 3 y exactly: the lags of x are aliased with those of y (R: grangertest stops with
    /// "there are aliased coefficients in the model").
    #[test]
    fn test_granger_rejects_proportional_series() {
        let y: Vec<f64> = (0..30)
            .map(|i| ((i * 7919) % 97) as f64 / 97.0 - 0.5)
            .collect();
        let x: Vec<f64> = y.iter().map(|v| 3.0 * v).collect();
        let ds = Dataset::new(df! { "y" => y, "x" => x }.unwrap());
        match granger_test(&ds, "y", "x", 2) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(
                    context.contains("unrestricted") && context.contains("x_1"),
                    "{context}"
                )
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
    }

    /// Simulated noisy pair: y depends on its own lag and (with weight `beta`) on x's.
    fn simulated_pair(n: usize, beta: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let (mut y, mut x) = (vec![0.0f64; n], vec![0.0f64; n]);
        for t in 1..n {
            let (e, u): (f64, f64) = (
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            );
            x[t] = 0.5 * x[t - 1] + u;
            y[t] = 0.3 * y[t - 1] + beta * x[t - 1] + e;
        }
        (y, x)
    }

    /// The normal-equations RSS pair (OLS's decision) agrees with the one-QR effects it
    /// replaces on well-conditioned designs, also when x adds almost nothing (small gain).
    #[test]
    fn test_granger_normal_equations_match_pivoted_qr() {
        for (n, beta, lags) in [
            (200usize, 0.4, 1usize),
            (200, 0.0, 4),
            (1000, 0.02, 6),
            (40, 0.3, 3),
        ] {
            let (yv, xv) = simulated_pair(n, beta, n as u64 + lags as u64);
            let y = Array1::from_vec(yv[lags..].to_vec());
            let x = Array2::from_shape_fn((n - lags, 2 * lags + 1), |(i, j)| match j {
                0 => 1.0,
                j if j <= lags => yv[lags + i - j],
                j => xv[lags + i - (j - lags)],
            });
            let (ru, gain) = granger_normal_rss(&x, &y, lags)
                .unwrap()
                .expect("the design must take the normal equations");
            let (ru_qr, gain_qr) = granger_qr_rss(&x, &y, lags, "y", "x").unwrap();
            assert!(gain >= 0.0);
            assert!((ru / ru_qr - 1.0).abs() < 1e-12, "RSS_U {ru} vs {ru_qr}");
            assert!(
                (gain - gain_qr).abs() <= 1e-10 * gain_qr + 1e-13 * ru_qr,
                "n = {n}, lags = {lags}: gain {gain} vs {gain_qr}"
            );
        }
    }

    /// x = 3 y up to noise far below lm.fit's tolerance is still aliased (the normal
    /// equations leave the design to the QR, which names the lags).
    #[test]
    fn test_granger_rejects_nearly_proportional_series() {
        let (y, _) = simulated_pair(40, 0.0, 9);
        let x: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, v)| 3.0 * v + 1e-12 * (((i * 7919) % 97) as f64 - 48.0))
            .collect();
        let ds = Dataset::new(df! { "y" => y, "x" => x }.unwrap());
        match granger_test(&ds, "y", "x", 2) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(
                    context.contains("unrestricted") && context.contains("x_1"),
                    "{context}"
                )
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
    }

    /// R: lmtest::grangertest(prod ~ e, order = 2) on vars::Canada (case var_canada).
    #[test]
    fn test_validate_granger_canada_prod_on_e() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("var_canada");
        let r = granger_test(&ds, "prod", "e", 2).unwrap();
        Check::new(&case)
            .scalar("grangertest_prod_on_e_F", r.f_statistic, Tol::stat(r.n_obs))
            .scalar("grangertest_prod_on_e_p", r.p_value, Tol::p_value(r.n_obs))
            .scalar("grangertest_prod_on_e_df2", r.df2 as f64, Tol::exact())
            .finish();
    }

    #[test]
    fn test_granger_test_basic() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 2).unwrap();

        assert_eq!(result.lags, 2);
        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df1, 2);
        assert_eq!(result.dependent, "y");
        assert_eq!(result.cause, "x");
    }

    #[test]
    fn test_granger_test_with_lag_1() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 1).unwrap();

        assert_eq!(result.lags, 1);
        assert_eq!(result.df1, 1);
        assert!(result.n_obs < 30); // Lost some obs due to lagging
    }

    #[test]
    fn test_granger_test_bidirectional() {
        let dataset = create_granger_test_dataset();
        let (result1, result2) = granger_test_bidirectional(&dataset, "y", "x", 1).unwrap();

        // result1: x -> y
        assert_eq!(result1.dependent, "y");
        assert_eq!(result1.cause, "x");

        // result2: y -> x
        assert_eq!(result2.dependent, "x");
        assert_eq!(result2.cause, "y");
    }

    #[test]
    fn test_granger_test_rss_ordering() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 2).unwrap();

        // Unrestricted model should always have RSS <= restricted RSS
        assert!(result.rss_unrestricted <= result.rss_restricted + 1e-10);
    }

    #[test]
    fn test_granger_test_displays_correctly() {
        let dataset = create_granger_test_dataset();
        let result = granger_test(&dataset, "y", "x", 1).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Granger Causality Test"));
        assert!(display.contains("F Statistic"));
        assert!(display.contains("p-value"));
        assert!(display.contains("does not Granger-cause") || display.contains("Granger-causes"));
    }

    #[test]
    fn test_run_granger_test_default_lags() {
        let dataset = create_granger_test_dataset();
        let result = run_granger_test(&dataset, "y", "x").unwrap();

        // Should automatically select lags
        assert!(result.lags >= 1);
        assert!(result.f_statistic >= 0.0);
    }

    #[test]
    fn test_granger_test_insufficient_data() {
        // Create small dataset
        let df = df! {
            "y" => [1.0, 2.0, 3.0],
            "x" => [0.5, 1.0, 1.5]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Should fail with too many lags
        let result = granger_test(&dataset, "y", "x", 5);
        assert!(result.is_err());
    }
}
