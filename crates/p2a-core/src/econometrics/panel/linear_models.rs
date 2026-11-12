//! Linear panel data models: Fixed Effects and Random Effects.
//!
//! # Mathematical Background
//!
//! For panel data with observations yᵢₜ for entity i at time t:
//!
//! yᵢₜ = αᵢ + Xᵢₜ'β + εᵢₜ
//!
//! ## Fixed Effects (Within) Estimator
//!
//! The FE estimator demeans the data within each entity:
//!
//! (yᵢₜ - ȳᵢ) = (Xᵢₜ - X̄ᵢ)'β + (εᵢₜ - ε̄ᵢ)
//!
//! This eliminates time-invariant unobserved heterogeneity αᵢ.
//!
//! ## Random Effects (GLS) Estimator
//!
//! The RE estimator assumes αᵢ is uncorrelated with Xᵢₜ and uses quasi-demeaning
//! with a factor per entity:
//!
//! (yᵢₜ - θᵢȳᵢ) = (1-θᵢ)α + (Xᵢₜ - θᵢX̄ᵢ)'β + (εᵢₜ - θᵢε̄ᵢ)
//!
//! where θᵢ = 1 - √(σ²_ν / (Tᵢσ²_η + σ²_ν)), σ²_ν the idiosyncratic and σ²_η the
//! individual-effect variance. Both are the Swamy-Arora estimates, with the
//! Baltagi-Chang between regression on unbalanced panels, as
//! `plm::ercomp(method = "swar")`; on a balanced panel all θᵢ are equal.
//!
//! # References
//!
//! - Mundlak, Y. (1978). On the pooling of time series and cross section data.
//!   *Econometrica*, 46(1), 69-85. https://doi.org/10.2307/1913646
//!
//! - Swamy, P.A.V.B. & Arora, S.S. (1972). The exact finite sample properties of
//!   the estimators of coefficients in the error components regression models.
//!   *Econometrica*, 40(2), 261-275.
//!
//! - Baltagi, B.H. & Chang, Y.-J. (1994). Incomplete panels: A comparative study
//!   of alternative estimators for the unbalanced one-way error component
//!   regression model. *Journal of Econometrics*, 62(2), 67-89.
//!
//! - Baltagi, B.H. (2013). *Econometric Analysis of Panel Data* (5th ed.).
//!   Wiley. ISBN: 978-1118672327.
//!
//! R equivalent: `plm::plm()` with `model = "within"` or `model = "random"`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Columns, MissingSummary, NaPolicy};
use crate::econometrics::hdfe::screen_regressors;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::eig_symmetric;
use crate::linalg::solve::{
    LM_QR_TOL, LeastSquares, Qr, SingularPolicy, least_squares, sym_inverse,
};
use crate::regression::ols::{certainly_full_rank, normal_equations_inverse};
use crate::resource;
use crate::traits::estimator::{
    SignificanceLevel, chi_squared_p_value, f_test_p_value, normal_cdf, t_test_p_value,
};

use super::types::{
    ClusterAdjustment, PanelConfig, PanelMethod, PanelResult, PanelVcov, TestDistribution,
    f_test_df2,
};
use super::utils::{demean_by_entity, demean_matrix_by_entity, entity_codes};

/// y, the regressors and the entity id on the rows complete in all of them (plm
/// `na.omit`), after refusing an empty regressor list.
fn panel_columns(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<Columns> {
    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one regressor is required".to_string(),
        });
    }
    ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_cols)
        .factor(entity_col)
        .na(NaPolicy::Omit)
        .extract()
}

/// The missing-value note as the first warning, when rows were dropped.
fn warnings_with_note(missing: &MissingSummary) -> Vec<String> {
    missing.note().into_iter().collect()
}

/// Run Fixed Effects (within) panel estimation with the default [`PanelConfig`]
/// (classical covariance, as plm).
///
/// # Arguments
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `entity_col` - Column name for entity/individual identifier
pub fn run_fixed_effects(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<PanelResult> {
    run_fixed_effects_with(dataset, y_col, x_cols, entity_col, &PanelConfig::default())
}

/// Run Fixed Effects (within) panel estimation, as `plm::plm(model = "within")`.
///
/// The covariance is `config.vcov`: classical `s² (X̃'X̃)⁻¹` of the within model
/// (plm's default), or the entity-clustered (Arellano) sandwich built from the
/// demeaned regressors and the within residuals, as plm
/// `vcovHC(method = "arellano")` with `k = K` slopes. Inference is Student t with
/// `n - N - K` degrees of freedom, and the adjusted within R² is plm's
/// `1 - (1 - R²)(n - 1)/(n - N - K)`. `f_stat` follows the covariance, as plm's
/// `summary(fe, vcov = ...)`: the classical within F test on `(K, n - N - K)`
/// degrees of freedom, or under the Arellano covariance the Wald F `W/K` on
/// `(K, G - 1)` (plm `pwaldtest(test = "F")` with a cluster vcov, `df2adj`).
/// The clustered covariance has rank at most `G - 1`, so with `K >= G` slopes,
/// or whenever the slope covariance is singular, the F test is not computed:
/// `f_stat` and `f_p_value` are NaN and a warning says why.
///
/// Rows with a missing value (null or NaN) in `y`, a regressor or the entity are
/// dropped and reported in `missing` (plm `na.omit`); an infinite value is refused
/// (`NonFiniteValues`). Entities are numbered in R factor order of the ids.
///
/// Regressors without variation left after removing the entity means (a
/// time-invariant regressor), and the later of regressors that are collinear after
/// demeaning, are not estimated: they are named in `dropped` and a warning, as plm
/// drops them. With no regressor left the call is `InvalidSpecification`.
pub fn run_fixed_effects_with(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    config: &PanelConfig,
) -> EconResult<PanelResult> {
    // y, X (no intercept for FE) and the entity on the complete rows
    let cols = panel_columns(dataset, y_col, x_cols, entity_col)?;
    let missing = cols.missing().clone();
    let y = cols.numeric_array(y_col);
    let design = cols.matrix(x_cols, false)?;
    let x = design.data;
    let all_names = design.column_names;
    let n = y.len();
    let (entity_ids, n_groups) = entity_codes(cols.factor(entity_col));

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "At least 2 entities required for panel estimation".to_string(),
        });
    }

    // Demean y and X by entity
    let y_demeaned = demean_by_entity(&y, &entity_ids, n_groups);
    let x_demeaned_all = demean_matrix_by_entity(&x, &entity_ids, n_groups);

    // Regressors absorbed by the entity effects or collinear after them are not
    // estimated (plm drops them); the fit solves from the screen's factorization
    let mut warnings = warnings_with_note(&missing);
    let screen = screen_regressors(
        x.view(),
        x_demeaned_all,
        &all_names,
        &format!("'{entity_col}'"),
    )?;
    warnings.extend(screen.warnings.iter().cloned());
    let x_demeaned = &screen.x;
    let var_names: Vec<String> = screen.kept.iter().map(|&j| all_names[j].clone()).collect();
    let k = screen.kept.len();

    // Degrees of freedom: n - N - k (lost N entity dummies)
    let df = n
        .checked_sub(n_groups + k)
        .filter(|&d| d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: n_groups + k + 1,
            provided: n,
            context: "Not enough observations for degrees of freedom".to_string(),
        })?;

    // OLS on demeaned data: β = (X̃'X̃)^{-1} X̃'ỹ, from the screen's factorization
    // (OLS's decision: normal equations, pivoted QR when ill-conditioned)
    let beta = screen.coefficients(&y_demeaned)?;
    let xtx_inv = screen.unscaled_covariance()?;

    // Residuals from demeaned model
    let y_hat: Array1<f64> = x_demeaned.dot(&beta);
    let residuals = &y_demeaned - &y_hat;

    // Sum of squared residuals
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / df as f64;

    // Classical (homoskedastic) FE variance, matching R's
    // `plm(..., model = "within")` default: Var(β̂) = sigma^2 * (X̃'X̃)^{-1};
    // or the entity-clustered sandwich (plm vcovHC(method = "arellano")).
    let vcov = match config.vcov {
        PanelVcov::Classical => &xtx_inv * sigma2,
        PanelVcov::Arellano(adj) => {
            let scores = entity_scores(x_demeaned, &residuals, &entity_ids, n_groups)?;
            cluster_vcov(&xtx_inv, &scores, n, k, adj)?
        }
    };
    let std_errors: Vec<f64> = vcov.diag().mapv(|v: f64| v.max(0.0).sqrt()).to_vec();

    // R-squared (within)
    let y_mean_demeaned = y_demeaned.mean().unwrap_or(0.0);
    let sst: f64 = y_demeaned
        .iter()
        .map(|y| (y - y_mean_demeaned).powi(2))
        .sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted within R-squared, plm's r.squared(dfcor = TRUE)
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df as f64);

    // F test of the slopes with the chosen covariance, as plm's summary: the
    // classical within F, or the Wald F W/K on (K, G - 1) for a cluster vcov.
    // The entity scores sum to X~'e~ = 0, so the cluster meat has rank at most
    // G - 1 and the clustered covariance is singular when K > G - 1.
    let f_df2 = f_test_df2(config.vcov, df, n_groups);
    // An exact within fit has zero standard errors: t, p and F are not defined
    let exact_fit = ssr == 0.0;
    if exact_fit {
        warnings.push(format!(
            "the within fit of '{y_col}' is exact (SSR = 0): the standard errors are zero, so \
             t, p and F are not defined"
        ));
    }
    let (f_stat, f_p_value) = match config.vcov {
        _ if exact_fit => (f64::NAN, f64::NAN),
        PanelVcov::Classical => {
            let f = if k > 0 && ssr > 0.0 {
                (sst - ssr) / (k as f64) / (ssr / df as f64)
            } else {
                0.0
            };
            (f, f_test_p_value(f, k as f64, f_df2 as f64))
        }
        PanelVcov::Arellano(_) if k > n_groups - 1 => {
            warnings.push(format!(
                "F test not computed: with K = {k} slopes and G = {n_groups} entities the \
                 clustered covariance matrix of the slopes has rank at most G - 1 = {}, so it \
                 is singular",
                n_groups - 1
            ));
            (f64::NAN, f64::NAN)
        }
        PanelVcov::Arellano(_) => match wald_statistic(&beta.view(), &vcov.view())? {
            Some(w) => {
                let f = w / k as f64;
                (f, f_test_p_value(f, k as f64, f_df2 as f64))
            }
            None => {
                warnings.push(
                    "F test not computed: the clustered covariance matrix of the slopes is \
                     singular"
                        .to_string(),
                );
                (f64::NAN, f64::NAN)
            }
        },
    };

    // t-statistics and p-values
    let beta_vec: Vec<f64> = beta.to_vec();
    let t_stats: Vec<f64> = beta_vec
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { f64::NAN })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    let coef_arr = ndarray::Array1::from_vec(beta_vec.clone());
    let se_arr = ndarray::Array1::from_vec(std_errors.clone());

    Ok(PanelResult {
        method: PanelMethod::FixedEffects,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients: beta_vec,
        std_errors,
        t_stats,
        p_values,
        r_squared,
        adj_r_squared,
        f_stat,
        f_p_value,
        n_obs: n,
        n_groups,
        df,
        entity_var: entity_col.to_string(),
        significance,
        sigma_u: None,
        sigma_e: Some(sigma2.sqrt()),
        theta: None,
        vcov_type: config.vcov,
        test_distribution: TestDistribution::StudentT,
        wald_chi2: None,
        wald_df: None,
        wald_p_value: None,
        theta_by_group: Vec::new(),
        warnings,
        missing,
        dropped: screen.dropped,
        coef_arr,
        se_arr,
        residuals,
        vcov,
    })
}

/// Run Random Effects (GLS) panel estimation with the default [`PanelConfig`]
/// (classical covariance, as plm).
///
/// # Arguments
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the independent variable columns
/// * `entity_col` - Column name for entity/individual identifier
pub fn run_random_effects(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<PanelResult> {
    run_random_effects_with(dataset, y_col, x_cols, entity_col, &PanelConfig::default())
}

/// Run Random Effects (Swamy-Arora GLS) panel estimation, as
/// `plm::plm(model = "random")` (`ercomp(method = "swar")`, Baltagi-Chang on
/// unbalanced panels).
///
/// One code path serves balanced and unbalanced panels:
/// 0. Regressors exactly collinear with the intercept and the earlier regressors are
///    dropped (the later of each collinear set), named in `dropped` and a warning, as
///    plm drops aliased coefficients.
/// 1. σ²_ν from the within regression, `SSR_w / (n - N - K_w)`, where `K_w` counts
///    the regressors that vary within entities (a time-invariant regressor is not in
///    the within regression; plm's `ercomp`). It stays in the model, identified by
///    the between variation.
/// 2. The between regression on the observation-level group means (each entity
///    weighted by its T_i), its residual quadratic form `quad2` and the trace
///    term `M_η = n - tr[(Σ T_i z̄_i z̄_i')⁻¹ Σ T_i² z̄_i z̄_i']`;
///    `σ²_η = (quad2 - (N - r) σ²_ν) / M_η`, with `r` the rank of the between
///    design (K + 1 unless the entity means of the regressors are collinear,
///    for example when a regressor does not vary between entities; a warning
///    then says so). A negative σ²_η is set to 0 with a warning, as plm.
/// 3. `θ_i = 1 - sqrt(σ²_ν / (T_i σ²_η + σ²_ν))`, quasi-demeaning
///    `y*_it = y_it - θ_i ȳ_i`, `X*_it = X_it - θ_i X̄_i`, and OLS on the
///    transformed model.
///
/// The covariance is `config.vcov`: classical `s² (X*'X*)⁻¹` with `s²` the
/// transformed residual variance (plm's `vcov`), or the entity-clustered
/// (Arellano) sandwich. Inference is normal (z), the joint test of the slopes is
/// a Wald chi-squared with the chosen covariance, and R² is `cor(y*, X*b)²`, all
/// as plm's `summary`. `f_stat` is the Wald F form `W/K` (plm
/// `pwaldtest(test = "F")`) on `(K, n - K - 1)` degrees of freedom, or on
/// `(K, G - 1)` under the Arellano covariance (plm's `df2adj` for a cluster
/// vcov).
///
/// Rows with a missing value (null or NaN) in `y`, a regressor or the entity are
/// dropped and reported in `missing` (plm `na.omit`); an infinite value is refused
/// (`NonFiniteValues`). Entities are numbered in R factor order of the ids. Refused:
/// no regressor that varies within entities (plm's "empty model"), `n <= N + K_w` (no
/// within degrees of freedom for σ²_ν) and `N <= K + 1` (the between regression is
/// not estimable, as plm).
///
/// # References
/// - Swamy, P.A.V.B. & Arora, S.S. (1972). The exact finite sample properties of
///   the estimators of coefficients in the error components regression models.
///   *Econometrica*, 40(2), 261-275.
/// - Baltagi, B.H. & Chang, Y.-J. (1994). Incomplete panels: A comparative study
///   of alternative estimators for the unbalanced one-way error component
///   regression model. *Journal of Econometrics*, 62(2), 67-89.
pub fn run_random_effects_with(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    config: &PanelConfig,
) -> EconResult<PanelResult> {
    // y, the design with intercept (column 0, "(Intercept)"; the slopes are 1..)
    // and the entity on the complete rows
    let cols = panel_columns(dataset, y_col, x_cols, entity_col)?;
    let missing = cols.missing().clone();
    let y = cols.numeric_array(y_col);
    let design = cols.matrix(x_cols, true)?;
    let x_all = design.data;
    let all_names = design.column_names;
    let n = y.len();

    let (entity_ids, n_groups) = entity_codes(cols.factor(entity_col));

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "At least 2 entities required for panel estimation".to_string(),
        });
    }

    let mut warnings = warnings_with_note(&missing);

    // Step 0: raw collinearity. The later of exactly collinear columns of [1, X] is
    // dropped from every step (plm's aliased coefficients); the intercept is column 0,
    // so it is never the aliased one. The QR runs only when the design is not certainly
    // of full rank (`ols::certainly_full_rank`).
    let aliased = if certainly_full_rank(x_all.view())? {
        Vec::new()
    } else {
        Qr::decompose(x_all.view(), LM_QR_TOL)?.aliased()
    };
    let dropped: Vec<String> = aliased.iter().map(|&j| all_names[j].clone()).collect();
    if !dropped.is_empty() {
        let quoted: Vec<String> = dropped.iter().map(|d| format!("'{d}'")).collect();
        warnings.push(format!(
            "{} dropped: collinear with the other regressors (the later of the collinear \
             regressors is not estimated, as in plm)",
            quoted.join(", ")
        ));
    }
    let keep: Vec<usize> = (0..x_all.ncols())
        .filter(|j| !aliased.contains(j))
        .collect();
    let x = if aliased.is_empty() {
        x_all
    } else {
        x_all.select(Axis(1), &keep)
    };
    let var_names: Vec<String> = keep.iter().map(|&j| all_names[j].clone()).collect();
    let p = x.ncols(); // K + 1
    let k_s = p - 1; // K

    // Step 1: within regression on the entity-demeaned slopes -> sigma^2_nu. Only
    // the slopes that vary within entities enter it (plm's ercomp); the demeaned
    // intercept column is zero and is left out.
    let y_w = demean_by_entity(&y, &entity_ids, n_groups);
    let x_demeaned = demean_matrix_by_entity(&x, &entity_ids, n_groups);
    let within = screen_regressors(
        x.slice(s![.., 1..]),
        x_demeaned.slice(s![.., 1..]),
        &var_names[1..],
        &format!("'{entity_col}'"),
    )
    .map_err(|e| match e {
        EconError::InvalidSpecification { .. } => EconError::InvalidSpecification {
            message: "random effects: no regressor varies within entities, so the within \
                      regression that estimates sigma_e^2 is empty (plm stops with 'empty \
                      model')"
                .to_string(),
        },
        other => other,
    })?;
    let k_w = within.kept.len();
    if n <= n_groups + k_w {
        return Err(EconError::InsufficientData {
            required: n_groups + k_w + 1,
            provided: n,
            context: "random effects: the within regression that estimates sigma_e^2 needs \
                      more observations than entities plus slopes"
                .to_string(),
        });
    }
    if n_groups <= k_s + 1 {
        return Err(EconError::InsufficientData {
            required: k_s + 2,
            provided: n_groups,
            context: "random effects: the Swamy-Arora between regression needs more entities \
                      than coefficients (as plm)"
                .to_string(),
        });
    }
    let sigma2_nu = within.rss(&y_w)? / (n - n_groups - k_w) as f64;
    if !(sigma2_nu > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "'{y_col}' has no residual variation within entities (sigma_e^2 = 0), so the \
                 random-effects transformation is not defined"
            ),
        });
    }

    // Step 2: per-entity sums in one pass over the observations
    let what = "random effects group means";
    let mut t_i = resource::try_vec(n_groups, 0usize, what)?;
    let mut ybar = resource::try_vec(n_groups, 0.0f64, what)?;
    let mut zbar = resource::try_zeros2(n_groups, p, what)?;
    for (row, &g) in entity_ids.iter().enumerate() {
        t_i[g] += 1;
        ybar[g] += y[row];
        for j in 0..p {
            zbar[[g, j]] += x[[row, j]];
        }
    }

    // Between moments: A = Σ T_i z̄ z̄', c = Σ T_i z̄ ȳ, B2 = Σ T_i² z̄ z̄'
    let mut a = Array2::<f64>::zeros((p, p));
    let mut c = Array1::<f64>::zeros(p);
    let mut b2 = Array2::<f64>::zeros((p, p));
    for g in 0..n_groups {
        resource::checkpoint()?;
        let tg = t_i[g] as f64;
        ybar[g] /= tg;
        for j in 0..p {
            zbar[[g, j]] /= tg;
        }
        for j in 0..p {
            c[j] += tg * zbar[[g, j]] * ybar[g];
            for l in 0..p {
                let zz = zbar[[g, j]] * zbar[[g, l]];
                a[[j, l]] += tg * zz;
                b2[[j, l]] += tg * tg * zz;
            }
        }
    }

    // Step 3: between regression, rank-aware (collinear entity means, such as
    // a regressor without between variation, make A singular; plm then drops
    // the aliased column)
    let (a_inv, rank_b) = between_inverse(&a)?;
    if rank_b < p {
        warnings.push(format!(
            "the Swamy-Arora between regression is rank deficient (rank {rank_b} of {p}): the \
             entity means of the regressors are collinear (for example, a regressor that does \
             not vary between entities), so a generalized inverse was used (plm drops the \
             aliased regressor; on balanced panels the result is identical)"
        ));
    }
    let b_between = a_inv.dot(&c);
    let mut quad2 = 0.0;
    for g in 0..n_groups {
        resource::checkpoint()?;
        let fitted: f64 = (0..p).map(|j| zbar[[g, j]] * b_between[j]).sum();
        quad2 += t_i[g] as f64 * (ybar[g] - fitted).powi(2);
    }
    let m_eta = n as f64 - a_inv.dot(&b2).diag().sum();
    if m_eta.is_nan() || m_eta <= 0.0 {
        return Err(EconError::Computation(
            "random effects: non-positive Swamy-Arora trace term".to_string(),
        ));
    }
    let mut sigma2_eta = (quad2 - (n_groups - rank_b) as f64 * sigma2_nu) / m_eta;
    if sigma2_eta < 0.0 {
        sigma2_eta = 0.0;
        warnings.push(
            "estimated individual-effect variance is negative and was set to 0 (as plm); \
             random effects reduces to pooled OLS"
                .to_string(),
        );
    }

    // Step 4: theta_i per entity and the quasi-demeaned model
    let mut theta_g = resource::try_vec(n_groups, 0.0f64, what)?;
    for g in 0..n_groups {
        resource::checkpoint()?;
        theta_g[g] = if sigma2_eta > 0.0 {
            1.0 - (sigma2_nu / (t_i[g] as f64 * sigma2_eta + sigma2_nu)).sqrt()
        } else {
            0.0
        };
    }
    let mut y_q = resource::try_zeros1(n, "random effects transformed data")?;
    let mut x_q = resource::try_zeros2(n, p, "random effects transformed data")?;
    for (row, &g) in entity_ids.iter().enumerate() {
        let th = theta_g[g];
        y_q[row] = y[row] - th * ybar[g];
        for j in 0..p {
            x_q[[row, j]] = x[[row, j]] - th * zbar[[g, j]];
        }
    }

    // Step 5: OLS on the transformed model. X* c = 0 exactly when X c = 0
    // (average over t: (1 - theta_i) xbar_i'c = 0 with theta_i < 1, which holds
    // because sigma_e^2 > 0 was checked above), so its rank is that of the data with
    // the intercept, which step 0 made full; a rank below that is a numerical failure. OLS's solve
    // decision: the normal equations when well conditioned, the pivoted QR otherwise.
    let (bread, beta, fitted, residuals) = match normal_equations_inverse(x_q.view())? {
        Some(normal) => {
            let beta = normal.coefficients(x_q.view(), &y_q);
            let fitted = x_q.dot(&beta);
            let residuals = &y_q - &fitted;
            (normal.inverse, beta, fitted, residuals)
        }
        None => {
            let ls = least_squares(x_q.view(), y_q.view())?;
            if ls.rank < p {
                return Err(quasi_demeaned_rank_error(&ls, p, &var_names));
            }
            (ls.unscaled_cov, ls.coefficients, ls.fitted, ls.residuals)
        }
    };
    let df = n
        .checked_sub(p)
        .filter(|&d| d > 0)
        .ok_or_else(|| EconError::InsufficientData {
            required: p + 1,
            provided: n,
            context: "random effects: residual degrees of freedom".to_string(),
        })?;
    let s2 = residuals.dot(&residuals) / df as f64;

    let vcov = match config.vcov {
        PanelVcov::Classical => &bread * s2,
        PanelVcov::Arellano(adj) => {
            let scores = entity_scores(&x_q, &residuals, &entity_ids, n_groups)?;
            cluster_vcov(&bread, &scores, n, p, adj)?
        }
    };
    let std_errors: Vec<f64> = vcov.diag().mapv(|v: f64| v.max(0.0).sqrt()).to_vec();

    // z statistics and normal p-values (plm's summary for random models)
    let beta_vec: Vec<f64> = beta.to_vec();
    let t_stats: Vec<f64> = beta_vec
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&z| 2.0 * normal_cdf(-z.abs()))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // R-squared: cor(y*, X* b)^2 of the transformed model (plm's r.squared)
    let r_squared = squared_correlation(&y_q, &fitted);
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df as f64);

    // Wald test of the slopes with the chosen covariance (plm's summary) and
    // its F form W/K (plm's pwaldtest(test = "F")), whose df2 is G - 1 for a
    // cluster vcov (pwaldtest's df2adj)
    let f_df2 = f_test_df2(config.vcov, df, n_groups);
    let wald = wald_statistic(&beta.slice(s![1..]), &vcov.slice(s![1.., 1..]))?;
    let (wald_chi2, wald_df, wald_p_value, f_stat, f_p_value) = match wald {
        Some(w) => {
            let kf = k_s as f64;
            (
                Some(w),
                Some(k_s),
                Some(chi_squared_p_value(w, kf)),
                w / kf,
                f_test_p_value(w / kf, kf, f_df2 as f64),
            )
        }
        None => {
            warnings.push(
                "Wald test not computed: the slope covariance matrix is singular".to_string(),
            );
            (None, None, None, f64::NAN, f64::NAN)
        }
    };

    // theta: observation-weighted mean (plm's mean theta); per entity if unbalanced
    let theta = t_i
        .iter()
        .zip(&theta_g)
        .map(|(&t, &th)| t as f64 * th)
        .sum::<f64>()
        / n as f64;
    let balanced = t_i.iter().all(|&t| t == t_i[0]);
    let theta_by_group = if balanced { Vec::new() } else { theta_g };

    let coef_arr = Array1::from_vec(beta_vec.clone());
    let se_arr = Array1::from_vec(std_errors.clone());

    Ok(PanelResult {
        method: PanelMethod::RandomEffects,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients: beta_vec,
        std_errors,
        t_stats,
        p_values,
        r_squared,
        adj_r_squared,
        f_stat,
        f_p_value,
        n_obs: n,
        n_groups,
        df,
        entity_var: entity_col.to_string(),
        significance,
        sigma_u: Some(sigma2_eta.sqrt()),
        sigma_e: Some(sigma2_nu.sqrt()),
        theta: Some(theta),
        vcov_type: config.vcov,
        test_distribution: TestDistribution::Normal,
        wald_chi2,
        wald_df,
        wald_p_value,
        theta_by_group,
        warnings,
        missing,
        dropped,
        coef_arr,
        se_arr,
        residuals,
        vcov,
    })
}

/// `b' V⁻¹ b`, or `None` when `V` is singular.
///
/// The rank is judged as in [`scaled_eigen`], on the unit-diagonal scaling
/// `S V S` against `100 p ε max(1, λ_max)`, and not by which branch an inverse
/// takes: Cholesky accepts an exactly singular positive semidefinite matrix
/// whose last pivot rounds to a tiny positive number (an entity-clustered
/// covariance with more coefficients than entities less one is such a
/// matrix). At full rank the statistic comes from the same eigen-pairs,
/// `Σ_j (u_j' S b)² / λ_j`, which is `b' V⁻¹ b` because
/// `V⁻¹ = S (S V S)⁻¹ S`.
///
/// `Ok(None)` when the covariance is singular; an eigen-decomposition failure is an
/// error.
fn wald_statistic(b: &ArrayView1<f64>, v: &ArrayView2<f64>) -> EconResult<Option<f64>> {
    let p = v.nrows();
    if p == 0 || v.ncols() != p || b.len() != p {
        return Ok(None);
    }
    let sym = Array2::from_shape_fn((p, p), |(i, j)| 0.5 * (v[[i, j]] + v[[j, i]]));
    let e = scaled_eigen(&sym)?;
    if e.rank < p {
        return Ok(None);
    }
    let sb: Vec<f64> = (0..p).map(|i| e.s[i] * b[i]).collect();
    let w: f64 = e
        .lambda
        .iter()
        .enumerate()
        .map(|(j, &l)| {
            let proj: f64 = (0..p).map(|i| e.u[[i, j]] * sb[i]).sum();
            proj * proj / l
        })
        .sum();
    Ok(w.is_finite().then_some(w))
}

/// The error of a quasi-demeaned random-effects design that the pivoted QR finds rank
/// deficient (step 5), naming the aliased regressors.
fn quasi_demeaned_rank_error(ls: &LeastSquares, p: usize, names: &[String]) -> EconError {
    EconError::SingularMatrix {
        context: format!(
            "random effects: quasi-demeaned design rank {} of {p} after the collinear \
             regressors were dropped",
            ls.rank
        ),
        suggestion: ls.aliasing_warning(names).unwrap_or_else(|| {
            "check the regressors for near-collinearity after the random-effects \
             transformation"
                .to_string()
        }),
    }
}

/// The unit-diagonal scaling `S A S`, `S = diag(A_jj^-1/2)` (0 for a zero
/// diagonal entry), of a symmetric positive semidefinite `A`, its eigen-pairs,
/// and its rank judged against `100 p ε max(1, λ_max)`, so that regressor units
/// do not matter.
struct ScaledEigen {
    s: Vec<f64>,
    lambda: Vec<f64>,
    u: Array2<f64>,
    tol: f64,
    rank: usize,
}

fn scaled_eigen(a: &Array2<f64>) -> EconResult<ScaledEigen> {
    let p = a.nrows();
    let s: Vec<f64> = (0..p)
        .map(|j| {
            let d = a[[j, j]];
            if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }
        })
        .collect();
    let a_s = Array2::from_shape_fn((p, p), |(i, j)| s[i] * a[[i, j]] * s[j]);
    let (lambda, u) = symmetric_eigen_pairs(&a_s)?;
    let lambda_max = lambda.iter().copied().fold(1.0, f64::max);
    let tol = 100.0 * p as f64 * f64::EPSILON * lambda_max;
    let rank = lambda.iter().filter(|&&l| l > tol).count();
    Ok(ScaledEigen {
        s,
        lambda,
        u,
        tol,
        rank,
    })
}

/// `cor(a, b)²` from centered sums; 0 when either has no variance.
fn squared_correlation(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let (ma, mb) = (a.mean().unwrap_or(0.0), b.mean().unwrap_or(0.0));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let (da, db) = (ai - ma, bi - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa > 0.0 && sbb > 0.0 {
        sab * sab / (saa * sbb)
    } else {
        0.0
    }
}

/// Eigen-decomposition of a symmetric matrix with each eigenvalue taken as the
/// Rayleigh quotient `u_j' M u_j` of its eigenvector, so that values and vectors
/// pair up whatever order the two underlying faer calls return.
pub(super) fn symmetric_eigen_pairs(m: &Array2<f64>) -> EconResult<(Vec<f64>, Array2<f64>)> {
    let (_, u) = eig_symmetric(&m.view())?;
    let lambda = (0..u.ncols())
        .map(|j| {
            let col = u.column(j);
            col.dot(&m.dot(&col))
        })
        .collect();
    Ok((lambda, u))
}

/// Inverse of the between moment matrix `A = Σ T_i z̄_i z̄_i'` and its rank.
///
/// The rank is judged as in [`scaled_eigen`], on `S A S` against
/// `100 (K + 1) ε max(1, λ_max)`. At full rank this is `sym_inverse(A)` with
/// `SingularPolicy::Error` (its rank tolerance is the lower one); otherwise the generalized
/// inverse `S (Σ_{λ_j > tol} u_j u_j' / λ_j) S`. Any generalized inverse gives
/// the same Swamy-Arora `quad2` and trace term, because the range of
/// `Σ T_i² z̄ z̄'` lies in the range of `A`.
fn between_inverse(a: &Array2<f64>) -> EconResult<(Array2<f64>, usize)> {
    let p = a.nrows();
    let ScaledEigen {
        s,
        lambda,
        u,
        tol,
        rank,
    } = scaled_eigen(a)?;
    if rank == p {
        // full rank was verified above
        let inv = sym_inverse(
            a.view(),
            SingularPolicy::Error,
            "Swamy-Arora between cross-product",
        )?
        .inverse;
        return Ok((inv, p));
    }
    let mut g = Array2::<f64>::zeros((p, p));
    for (j, &l) in lambda.iter().enumerate() {
        if l > tol {
            let col = u.column(j);
            for r in 0..p {
                for c in 0..p {
                    g[[r, c]] += col[r] * col[c] / l;
                }
            }
        }
    }
    Ok((
        Array2::from_shape_fn((p, p), |(i, j)| s[i] * g[[i, j]] * s[j]),
        rank,
    ))
}

/// Per-entity scores `g_i = Σ_t x_it e_it` (an `N × k` matrix).
fn entity_scores(
    x: &Array2<f64>,
    e: &Array1<f64>,
    entity_ids: &[usize],
    n_groups: usize,
) -> EconResult<Array2<f64>> {
    let k = x.ncols();
    let mut scores = resource::try_zeros2(n_groups, k, "entity-clustered scores")?;
    for (row, &g) in entity_ids.iter().enumerate() {
        for j in 0..k {
            scores[[g, j]] += x[[row, j]] * e[row];
        }
    }
    Ok(scores)
}

/// Entity-clustered (Arellano) covariance `c · B M B`, `M = Σ_i g_i g_i'`, for
/// `bread` B, per-entity `scores` g_i, `n` observations and `k` coefficients.
/// `c` is 1 (HC0), `n/(n - k)` (HC1) or `G/(G - 1) (n - 1)/(n - k)` (Stata), as
/// plm `vcovHC(method = "arellano", type = "HC0" | "HC1" | "sss")`.
fn cluster_vcov(
    bread: &Array2<f64>,
    scores: &Array2<f64>,
    n: usize,
    k: usize,
    adj: ClusterAdjustment,
) -> EconResult<Array2<f64>> {
    let n_clusters = scores.nrows();
    resource::checkpoint()?;
    let meat = scores.t().dot(scores);
    let (n, k, g) = (n as f64, k as f64, n_clusters as f64);
    let factor = match adj {
        ClusterAdjustment::HC0 => 1.0,
        ClusterAdjustment::HC1 => n / (n - k),
        ClusterAdjustment::Stata => g / (g - 1.0) * (n - 1.0) / (n - k),
    };
    Ok(bread.dot(&meat).dot(bread) * factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    /// A budget whose token is already cancelled.
    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// A balanced synthetic panel of `n_ent` entities and `n_t` periods (the
    /// same generator as `gls_models::tests::synthetic_panel`).
    fn synthetic_panel(n_ent: usize, n_t: usize) -> Dataset {
        let n = n_ent * n_t;
        let entity: Vec<i64> = (0..n).map(|i| (i / n_t) as i64).collect();
        let time: Vec<i64> = (0..n).map(|i| (i % n_t) as i64 + 2000).collect();
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64 / 10.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                1.0 + 0.5 * x[i]
                    + ((i / n_t) % 17) as f64 * 0.1
                    + ((i * 7919) % 1009) as f64 / 1009.0
                    - 0.5
            })
            .collect();
        Dataset::new(
            polars::prelude::df! { "entity" => entity, "time" => time, "x" => x, "y" => y }
                .unwrap(),
        )
    }

    #[test]
    fn test_random_effects_scaling_many_entities() {
        if cfg!(debug_assertions) {
            return; // release-only timing check
        }
        let ds = synthetic_panel(100_000, 5);
        let t0 = std::time::Instant::now();
        let r = run_random_effects(&ds, "y", &["x"], "entity").unwrap();
        assert_eq!(r.n_groups, 100_000);
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(10),
            "took {:?}",
            t0.elapsed()
        );
    }

    /// `n - N - K` within degrees of freedom: 8 rows, 6 entities, 2 slopes.
    #[test]
    fn test_random_effects_needs_within_degrees_of_freedom() {
        let ds = Dataset::new(
            polars::prelude::df! {
                "entity" => [1i64, 1, 2, 2, 3, 4, 5, 6],
                "x1" => [1.0, 2.5, 0.3, 1.9, 3.1, 0.7, 2.2, 1.4],
                "x2" => [0.4, -0.2, 1.1, 0.8, -0.6, 0.9, 0.1, -1.3],
                "y" => [2.3, 4.1, 1.2, 3.8, 4.4, 1.9, 3.0, 0.6],
            }
            .unwrap(),
        );
        match run_random_effects(&ds, "y", &["x1", "x2"], "entity") {
            Err(EconError::InsufficientData { context, .. }) => {
                assert!(context.contains("within"), "{context}")
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    /// The between regression needs N > K + 1 entities (as plm): N = 3, K = 2.
    #[test]
    fn test_random_effects_needs_more_entities_than_coefficients() {
        let ds = Dataset::new(
            polars::prelude::df! {
                "entity" => [1i64, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                "x1" => [1.0, 2.5, 0.3, 1.9, 3.1, 0.7, 2.2, 1.4, 0.9, 2.8, 1.6, 0.2],
                "x2" => [0.4, -0.2, 1.1, 0.8, -0.6, 0.9, 0.1, -1.3, 0.5, -0.4, 1.2, 0.3],
                "y" => [2.3, 4.1, 1.2, 3.8, 4.4, 1.9, 3.0, 0.6, 2.7, 4.9, 3.3, 1.1],
            }
            .unwrap(),
        );
        match run_random_effects(&ds, "y", &["x1", "x2"], "entity") {
            Err(EconError::InsufficientData { context, .. }) => {
                assert!(context.contains("between"), "{context}")
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    /// Classical FE has no budget check here (its demeaning lives in `utils`);
    /// the Arellano path checks the budget before the cluster meat.
    #[test]
    fn test_fixed_effects_cluster_stops_when_cancelled() {
        let ds = synthetic_panel(50, 4);
        let config = PanelConfig {
            vcov: PanelVcov::Arellano(ClusterAdjustment::Stata),
        };
        let res = resource::with_budget(cancelled(), || {
            run_fixed_effects_with(&ds, "y", &["x"], "entity", &config)
        });
        assert!(matches!(res, Err(EconError::Cancelled(_))), "{res:?}");
    }

    #[test]
    fn test_random_effects_stops_when_cancelled() {
        let ds = synthetic_panel(50, 4);
        let res = resource::with_budget(cancelled(), || {
            run_random_effects(&ds, "y", &["x"], "entity")
        });
        assert!(matches!(res, Err(EconError::Cancelled(_))), "{res:?}");
    }

    /// `synthetic_panel(10, 4)` with `y[row]` or `x[row]` replaced by `value`.
    fn panel_with(column: &str, row: usize, value: f64) -> Dataset {
        let df = synthetic_panel(10, 4).df().clone();
        let mut v: Vec<f64> = df
            .column(column)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        v[row] = value;
        let mut df = df;
        df.with_column(polars::prelude::Column::new(column.into(), v))
            .unwrap();
        Dataset::new(df)
    }

    /// The column and count of a `NonFiniteValues` error.
    fn non_finite(res: EconResult<PanelResult>) -> (String, usize) {
        match res {
            Err(EconError::NonFiniteValues { column, count }) => (column, count),
            other => panic!("expected NonFiniteValues, got {other:?}"),
        }
    }

    /// `synthetic_panel(10, 4)` without row `row`.
    fn panel_without(row: usize) -> Dataset {
        let df = synthetic_panel(10, 4).df().clone();
        use polars::prelude::{BooleanChunked, NewChunkedArray};
        let keep: Vec<bool> = (0..df.height()).map(|i| i != row).collect();
        let mask = BooleanChunked::from_slice("keep".into(), &keep);
        Dataset::new(df.filter(&mask).unwrap())
    }

    /// Coefficients and SEs of two fits agree to 1e-12.
    fn assert_same_fit(a: &PanelResult, b: &PanelResult) {
        let tol = p2a_validation::Tol::new(1e-12, 1e-12);
        assert_eq!(a.variables, b.variables);
        for (x, y) in a
            .coefficients
            .iter()
            .chain(&a.std_errors)
            .zip(b.coefficients.iter().chain(&b.std_errors))
        {
            assert!(tol.allows(*x, *y), "{x} vs {y}");
        }
    }

    /// A NaN in y is a missing cell (plm's na.omit): the row is dropped and
    /// reported, and the fit equals the fit without that row; an infinite regressor
    /// value is refused by name.
    #[test]
    fn test_fixed_effects_non_finite_data() {
        let classical = PanelConfig::default();
        let arellano = PanelConfig {
            vcov: PanelVcov::Arellano(ClusterAdjustment::Stata),
        };
        for config in [classical, arellano] {
            let fe = |ds: &Dataset| run_fixed_effects_with(ds, "y", &["x"], "entity", &config);
            let r = fe(&panel_with("y", 5, f64::NAN)).unwrap();
            assert_same_fit(&r, &fe(&panel_without(5)).unwrap());
            assert_eq!(r.missing.n_dropped, 1);
            assert_eq!(r.missing.by_column, [("y".to_string(), 1)]);
            assert_eq!(r.n_obs, 39);
        }
        assert_eq!(
            non_finite(run_fixed_effects(
                &panel_with("x", 7, f64::INFINITY),
                "y",
                &["x"],
                "entity"
            )),
            ("x".to_string(), 1)
        );
    }

    /// RE drops a NaN row as FE does; a negative infinity is refused by name.
    #[test]
    fn test_random_effects_non_finite_data() {
        let re = |ds: &Dataset| run_random_effects(ds, "y", &["x"], "entity");
        let r = re(&panel_with("y", 5, f64::NAN)).unwrap();
        assert_same_fit(&r, &re(&panel_without(5)).unwrap());
        assert_eq!(r.missing.n_dropped, 1);
        assert_eq!(r.missing.by_column, [("y".to_string(), 1)]);
        assert_eq!(
            non_finite(re(&panel_with("x", 7, f64::NEG_INFINITY))),
            ("x".to_string(), 1)
        );
    }

    /// FE and RE on Grunfeld with missing cells and two missing ids, pinned to plm on
    /// the 195 rows complete in (inv, value, capital, id), for a text and a numeric id.
    #[test]
    fn test_validate_panel_fe_re_grunfeld_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        for id in ["firm_name", "firm_id"] {
            let fe = run_fixed_effects(&ds, "inv", &["value", "capital"], id).unwrap();
            let re = run_random_effects(&ds, "inv", &["value", "capital"], id).unwrap();
            let n = fe.n_obs;
            let by = |c: &str| {
                fe.missing
                    .by_column
                    .iter()
                    .find(|(k, _)| k == c)
                    .map_or(0, |(_, v)| *v) as f64
            };
            Check::new(&case)
                .named("fe_coef", &fe.variables, &fe.coefficients, Tol::coef(n))
                .named("fe_se", &fe.variables, &fe.std_errors, Tol::se(n))
                .scalar("fe_nobs", fe.n_obs as f64, Tol::exact())
                .named("re_coef", &re.variables, &re.coefficients, Tol::coef(n))
                .named("re_se", &re.variables, &re.std_errors, Tol::se(n))
                .scalar("re_nobs", re.n_obs as f64, Tol::exact())
                .scalar("n_groups", fe.n_groups as f64, Tol::exact())
                .scalar("missing_inv", by("inv"), Tol::exact())
                .scalar("missing_value", by("value"), Tol::exact())
                // firm_id is missing in the same 2 rows as firm_name
                .scalar("missing_firm_name", by(id), Tol::exact())
                .finish();
            let note = format!(
                "5 of 200 rows dropped because of missing values (inv: 2, value: 1, {id}: 2)"
            );
            assert_eq!(fe.missing.note(), Some(note.clone()));
            assert_eq!(re.missing, fe.missing);
            assert!(fe.warnings.contains(&note) && re.warnings.contains(&note));
            let text = fe.to_string();
            assert_eq!(text.matches(&note).count(), 1, "{text}");
            assert!(
                text.trim_end()
                    .ends_with(&format!("Missing values: {note}")),
                "{text}"
            );
            assert_eq!(re.n_groups, 10);
        }
    }

    /// ECON-37: with y and x complete, two missing ids used to become entity 0 (numeric,
    /// merged into the real firm 0) or the pseudo-entity "" (text, an 11th group). They
    /// now drop their rows, so both id columns give plm's estimate on 195 rows.
    #[test]
    fn test_validate_panel_fe_missing_ids_not_merged() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        let subset = [
            "inv".to_string(),
            "value".to_string(),
            "capital".to_string(),
        ];
        let xy = Dataset::new(ds.df().drop_nulls(Some(&subset)).unwrap());
        assert_eq!(xy.df().height(), 197);
        for id in ["firm_name", "firm_id"] {
            let fe = run_fixed_effects(&xy, "inv", &["value", "capital"], id).unwrap();
            let re = run_random_effects(&xy, "inv", &["value", "capital"], id).unwrap();
            Check::new(&case)
                .named("fe_coef", &fe.variables, &fe.coefficients, Tol::coef(195))
                .named("fe_se", &fe.variables, &fe.std_errors, Tol::se(195))
                .named("re_coef", &re.variables, &re.coefficients, Tol::coef(195))
                .named("re_se", &re.variables, &re.std_errors, Tol::se(195))
                .finish();
            for r in [&fe, &re] {
                assert_eq!(
                    (r.n_groups, r.n_obs, r.missing.n_dropped),
                    (10, 195, 2),
                    "{id}"
                );
            }
        }
    }

    /// The errors name the column and its problem; an Int64 regressor gives the fit of
    /// the same values in Float64.
    #[test]
    fn test_validate_na_fe_re_errors_and_integer_regressor() {
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        let non_numeric = |r: EconResult<PanelResult>| match r {
            Err(EconError::NonNumericColumn { column }) => column,
            other => panic!("expected NonNumericColumn, got {other:?}"),
        };
        assert_eq!(
            non_numeric(run_fixed_effects(&ds, "firm_name", &["value"], "firm")),
            "firm_name"
        );
        assert_eq!(
            non_numeric(run_random_effects(&ds, "inv", &["firm_name"], "firm")),
            "firm_name"
        );
        match run_fixed_effects(&ds, "inv", &["value"], "nope") {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        for r in [
            run_fixed_effects(&ds, "inv", &[], "firm"),
            run_random_effects(&ds, "inv", &[], "firm"),
        ] {
            match r {
                Err(EconError::InvalidSpecification { message }) => {
                    assert_eq!(message, "At least one regressor is required")
                }
                other => panic!("expected InvalidSpecification, got {other:?}"),
            }
        }

        // Grunfeld's Int64 year as a regressor equals the same values in Float64
        let path = p2a_validation::repo_path(&case.datasets[0]);
        let g = Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap());
        assert_eq!(
            g.df().column("year").unwrap().dtype(),
            &polars::prelude::DataType::Int64
        );
        let mut df = g.df().clone();
        let year_f = df
            .column("year")
            .unwrap()
            .cast(&polars::prelude::DataType::Float64)
            .unwrap();
        df.with_column(year_f).unwrap();
        let g_f = Dataset::new(df);
        let x = ["value", "year"];
        assert_same_fit(
            &run_fixed_effects(&g, "inv", &x, "firm").unwrap(),
            &run_fixed_effects(&g_f, "inv", &x, "firm").unwrap(),
        );
        assert_same_fit(
            &run_random_effects(&g, "inv", &x, "firm").unwrap(),
            &run_random_effects(&g_f, "inv", &x, "firm").unwrap(),
        );
    }

    /// A noisy panel of `g` entities and `t` periods with regressors `x1..xk`
    /// and an entity effect in `y`, all from a splitmix64 stream started at
    /// `seed`.
    fn small_noisy_panel(g: usize, t: usize, k: usize, seed: u64) -> (Dataset, Vec<String>) {
        let mut state = seed;
        let mut uniform = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let n = g * t;
        let entity: Vec<i64> = (0..n).map(|i| (i / t) as i64).collect();
        let xs: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..n).map(|_| uniform()).collect())
            .collect();
        let effects: Vec<f64> = (0..g).map(|_| uniform()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                1.0 + xs.iter().map(|x| 0.5 * x[i]).sum::<f64>() + effects[i / t] + 0.3 * uniform()
            })
            .collect();
        let names: Vec<String> = (1..=k).map(|j| format!("x{j}")).collect();
        let mut columns = vec![
            polars::prelude::Column::new("entity".into(), entity),
            polars::prelude::Column::new("y".into(), y),
        ];
        for (name, x) in names.iter().zip(xs) {
            columns.push(polars::prelude::Column::new(name.as_str().into(), x));
        }
        (
            Dataset::new(polars::prelude::DataFrame::new(columns).unwrap()),
            names,
        )
    }

    /// Twelve 2-entity panels with 2 regressors (T = 5 or 6) and twelve
    /// 3-entity panels with 3 regressors: K >= G in each.
    fn panels_with_k_at_least_g() -> Vec<(Dataset, Vec<String>)> {
        let mut panels = Vec::new();
        for g in [2, 3] {
            for seed in 0..12u64 {
                panels.push(small_noisy_panel(
                    g,
                    5 + (seed % 2) as usize,
                    g,
                    1000 * g as u64 + seed,
                ));
            }
        }
        panels
    }

    /// Under the entity-clustered covariance the FE scores sum to
    /// `X~'e~ = 0`, so the cluster meat has rank at most `G - 1`, and with
    /// `K >= G` slopes the clustered slope covariance is singular. Cholesky
    /// still accepted some of these matrices (a last pivot that rounds to a
    /// tiny positive number) and the Wald F came out as 1e15 or more, highly
    /// significant on noise; plm's `pwaldtest` usually stops there ('system is
    /// computationally singular' or 'exactly singular') and otherwise returns a
    /// meaningless F of order 1e17. The F test is now not computed, with a warning
    /// naming K and G, and Display has no F line.
    #[test]
    fn test_fixed_effects_cluster_f_not_computed_when_k_reaches_g() {
        let config = PanelConfig {
            vcov: PanelVcov::Arellano(ClusterAdjustment::Stata),
        };
        for (ds, names) in panels_with_k_at_least_g() {
            let x: Vec<&str> = names.iter().map(String::as_str).collect();
            let r = run_fixed_effects_with(&ds, "y", &x, "entity", &config).unwrap();
            let (k, g) = (x.len(), r.n_groups);
            assert!(
                r.f_stat.is_nan() && r.f_p_value.is_nan(),
                "G = {g}, K = {k}: F {} (p {})",
                r.f_stat,
                r.f_p_value
            );
            let expected = format!("K = {k} slopes and G = {g} entities");
            assert!(
                r.warnings
                    .iter()
                    .any(|w| w.starts_with("F test not computed") && w.contains(&expected)),
                "{:?}",
                r.warnings
            );
            assert!(!r.to_string().contains("F-statistic"), "{r}");
        }
    }

    /// The rank of the covariance, not the branch the inverse takes,
    /// decides whether a Wald statistic exists: every rank-deficient
    /// clustered FE covariance above gives `None`, including those that
    /// Cholesky accepted.
    #[test]
    fn test_wald_statistic_refuses_a_singular_covariance() {
        let config = PanelConfig {
            vcov: PanelVcov::Arellano(ClusterAdjustment::Stata),
        };
        for (ds, names) in panels_with_k_at_least_g() {
            let x: Vec<&str> = names.iter().map(String::as_str).collect();
            let r = run_fixed_effects_with(&ds, "y", &x, "entity", &config).unwrap();
            let w = wald_statistic(&r.coef_arr.view(), &r.vcov.view()).unwrap();
            assert!(w.is_none(), "G = {}, K = {}: {w:?}", r.n_groups, x.len());
        }
        // A zero variance is singular whatever the other entries
        let v = Array2::from_shape_vec((2, 2), vec![4.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(
            wald_statistic(&Array1::from(vec![1.0, 2.0]).view(), &v.view())
                .unwrap()
                .is_none()
        );
    }

    /// The degrees of freedom `PanelResult::f_df` reports are those the
    /// estimators computed `f_p_value` on (one rule, `f_test_df2`), for both
    /// estimators and both covariances.
    #[test]
    fn test_f_df_are_those_of_the_f_p_value() {
        let ds = synthetic_panel(10, 4);
        for vcov in [
            PanelVcov::Classical,
            PanelVcov::Arellano(ClusterAdjustment::Stata),
        ] {
            let config = PanelConfig { vcov };
            let fe = run_fixed_effects_with(&ds, "y", &["x"], "entity", &config).unwrap();
            let re = run_random_effects_with(&ds, "y", &["x"], "entity", &config).unwrap();
            for r in [fe, re] {
                let (df1, df2) = r.f_df();
                assert!(r.f_stat.is_finite(), "{vcov:?}: {}", r.f_stat);
                assert_eq!(
                    r.f_p_value,
                    f_test_p_value(r.f_stat, df1 as f64, df2 as f64),
                    "{vcov:?} {:?}",
                    r.method
                );
                let want = match vcov {
                    PanelVcov::Classical => r.df,
                    PanelVcov::Arellano(_) => 9,
                };
                assert_eq!((df1, df2), (1, want), "{vcov:?} {:?}", r.method);
            }
        }
    }

    /// `b' V^-1 b` for an invertible `V`, also after rescaling a regressor
    /// by 1e7 (the rank decision does not depend on units): 23/11.
    #[test]
    fn test_wald_statistic_full_rank() {
        let b = Array1::from(vec![1.0, -2.0]);
        let v = Array2::from_shape_vec((2, 2), vec![4.0, 1.0, 1.0, 3.0]).unwrap();
        let w = wald_statistic(&b.view(), &v.view()).unwrap().unwrap();
        assert!((w - 23.0 / 11.0).abs() <= 1e-14, "{w}");
        let b = Array1::from(vec![1.0, -2e-7]);
        let v = Array2::from_shape_vec((2, 2), vec![4.0, 1e-7, 1e-7, 3e-14]).unwrap();
        let w = wald_statistic(&b.view(), &v.view()).unwrap().unwrap();
        assert!((w - 23.0 / 11.0).abs() <= 1e-13, "{w}");
    }

    /// The cluster meat as one product `G'G` equals the sum of the per-entity
    /// outer products `g_i g_i'`.
    #[test]
    fn test_cluster_vcov_meat_is_sum_of_outer_products() {
        let scores =
            Array2::from_shape_vec((4, 2), vec![0.3, -1.2, 2.1, 0.4, -0.7, 0.9, 1.6, -0.25])
                .unwrap();
        let bread = Array2::from_shape_vec((2, 2), vec![2.0, 0.3, 0.3, 1.5]).unwrap();
        let v = cluster_vcov(&bread, &scores, 20, 2, ClusterAdjustment::HC0).unwrap();
        let mut meat = Array2::<f64>::zeros((2, 2));
        for i in 0..4 {
            for a in 0..2 {
                for b in 0..2 {
                    meat[[a, b]] += scores[[i, a]] * scores[[i, b]];
                }
            }
        }
        let expected = bread.dot(&meat).dot(&bread);
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }
    }

    // ── T3-WP17: regressors absorbed by the entity effects or collinear (ECON-72)

    #[test]
    fn test_validate_fe_drops_time_invariant_regressor() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_absorbed_regressors");
        let n = 180;
        let r = run_fixed_effects(&ds, "y", &["x1", "x2", "z"], "id").unwrap();
        assert_eq!(r.dropped, vec!["z".to_string()]);
        assert!(
            r.warnings.iter().any(|w| w
                == "'z' dropped: no variation left after removing the 'id' fixed effects (not \
                    estimated, as in plm and fixest)"),
            "{:?}",
            r.warnings
        );
        Check::new(&case)
            .named("fe_coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("fe_se", &r.variables, &r.std_errors, Tol::se(n))
            .scalar("fe_df_resid", r.df as f64, Tol::exact())
            .finish();
        let r3 = run_fixed_effects(&ds, "y", &["x1", "x2", "x3"], "id").unwrap();
        assert_eq!(r3.dropped, vec!["x3".to_string()]);
        assert!(r3.warnings.iter().any(|w| w.starts_with(
            "'x3' dropped: collinear with the other regressors after removing the 'id' fixed \
             effects"
        )));
        Check::new(&case)
            .named("fe_x3_coef", &r3.variables, &r3.coefficients, Tol::coef(n))
            .finish();
        // only z: nothing left (plm "empty model"; fixest "All variables ... collinear")
        let e = run_fixed_effects(&ds, "y", &["z"], "id").unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.starts_with("no regressor left to estimate: 'z' dropped")),
            "{e:?}"
        );
    }

    #[test]
    fn test_validate_re_absorbed_regressor() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("panel_absorbed_regressors");
        let n = 180;
        let r = run_random_effects(&ds, "y", &["x1", "x2", "z"], "id").unwrap();
        assert!(r.dropped.is_empty()); // z is identified by the between variation
        Check::new(&case)
            .named("re_coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("re_se", &r.variables, &r.std_errors, Tol::se(n))
            .scalar("re_sigma2_idios", r.sigma_e.unwrap().powi(2), Tol::stat(n))
            .scalar("re_sigma2_id", r.sigma_u.unwrap().powi(2), Tol::stat(n))
            .scalar("re_theta", r.theta.unwrap(), Tol::stat(n))
            .finish();
        let r3 = run_random_effects(&ds, "y", &["x1", "x2", "x3"], "id").unwrap();
        assert_eq!(r3.dropped, vec!["x3".to_string()]);
        assert!(r3.warnings.iter().any(|w| {
            w.starts_with("'x3' dropped: collinear with the other regressors (the later")
        }));
        Check::new(&case)
            .named("re_x3_coef", &r3.variables, &r3.coefficients, Tol::coef(n))
            .finish();
        let e = run_random_effects(&ds, "y", &["z"], "id").unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("empty")),
            "{e:?}"
        );
    }

    /// A copy of `ds` with column `name` replaced by `values`.
    fn with_column(ds: &Dataset, name: &str, values: Vec<f64>) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(polars::prelude::Column::new(name.into(), values))
            .unwrap();
        Dataset::new(df)
    }

    /// The `wald_statistic` branch that finds the clustered slope covariance singular
    /// with K <= G - 1: `x2` varies only within entity 0, so the FE normal equations
    /// force entity 0's score of `x2` to zero and the meat has a zero row and column.
    #[test]
    fn test_fixed_effects_cluster_f_not_computed_when_slope_covariance_singular() {
        use crate::data::columns::{NaPolicy, numeric_column};
        let (ds, _) = small_noisy_panel(6, 5, 2, 77);
        let x2 = numeric_column(ds.df(), "x2", NaPolicy::Fail).unwrap().0;
        let x2: Vec<f64> = x2
            .iter()
            .enumerate()
            .map(|(i, &v)| if i < 5 { v } else { 0.25 * (i / 5) as f64 })
            .collect();
        let ds = with_column(&ds, "x2", x2);
        let config = PanelConfig {
            vcov: PanelVcov::Arellano(ClusterAdjustment::Stata),
        };
        let r = run_fixed_effects_with(&ds, "y", &["x1", "x2"], "entity", &config).unwrap();
        assert!(r.f_stat.is_nan() && r.f_p_value.is_nan(), "{r}");
        assert!(
            r.warnings.iter().any(|w| w
                == "F test not computed: the clustered covariance matrix of the slopes is \
                    singular"),
            "{:?}",
            r.warnings
        );
        assert!(
            r.std_errors.iter().all(|s| s.is_finite()),
            "{:?}",
            r.std_errors
        );
        assert!(!r.to_string().contains("F-statistic"), "{r}");
    }

    /// A panel whose outcome is constant within each entity (whole numbers, so the
    /// entity means are exact) with a noisy regressor.
    fn constant_within_panel() -> Dataset {
        let (ds, _) = small_noisy_panel(6, 5, 1, 91);
        let y: Vec<f64> = (0..30).map(|i| (3 * (i / 5) + 1) as f64).collect();
        with_column(&ds, "y", y)
    }

    #[test]
    fn test_fixed_effects_exact_within_fit_reports_undefined_inference() {
        let ds = constant_within_panel();
        for vcov in [
            PanelVcov::Classical,
            PanelVcov::Arellano(ClusterAdjustment::Stata),
        ] {
            let r =
                run_fixed_effects_with(&ds, "y", &["x1"], "entity", &PanelConfig { vcov }).unwrap();
            assert!(
                r.t_stats[0].is_nan() && r.p_values[0].is_nan(),
                "{vcov:?}: {:?} {:?}",
                r.t_stats,
                r.p_values
            );
            assert!(r.f_stat.is_nan() && r.f_p_value.is_nan(), "{vcov:?}");
            assert!(
                r.warnings
                    .iter()
                    .any(|w| w.starts_with("the within fit of 'y' is exact (SSR = 0)")),
                "{vcov:?}: {:?}",
                r.warnings
            );
        }
    }

    #[test]
    fn test_random_effects_refuses_no_within_variation() {
        let ds = constant_within_panel();
        match run_random_effects(&ds, "y", &["x1"], "entity") {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("'y'") && message.contains("sigma_e^2 = 0"),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// A rank-deficient quasi-demeaned design is a SingularMatrix whose suggestion names
    /// the aliased regressor.
    #[test]
    fn test_quasi_demeaned_rank_error_names_aliased_regressors() {
        let a = [0.3, 1.7, -0.4, 2.2, 0.9, -1.1];
        let x = Array2::from_shape_fn((6, 3), |(i, j)| match j {
            0 => 1.0,
            1 => a[i],
            _ => 2.0 * a[i],
        });
        let y = Array1::from(vec![1.2, 0.4, 2.9, -0.3, 1.8, 0.7]);
        let ls = least_squares(x.view(), y.view()).unwrap();
        assert_eq!(ls.rank, 2);
        let names = ["(Intercept)", "x1", "x2"].map(String::from);
        match quasi_demeaned_rank_error(&ls, 3, &names) {
            EconError::SingularMatrix {
                context,
                suggestion,
            } => {
                assert!(
                    context.contains("random effects: quasi-demeaned design rank 2 of 3"),
                    "{context}"
                );
                assert!(suggestion.contains("x2"), "{suggestion}");
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
    }

    /// A NaN F test (clustered FE with K >= G) is written as null and read back as NaN.
    #[test]
    fn test_panel_result_json_round_trips_nan_f() {
        let config = PanelConfig {
            vcov: PanelVcov::Arellano(ClusterAdjustment::Stata),
        };
        let (ds, names) = panels_with_k_at_least_g().remove(0);
        let x: Vec<&str> = names.iter().map(String::as_str).collect();
        let r = run_fixed_effects_with(&ds, "y", &x, "entity", &config).unwrap();
        assert!(r.f_stat.is_nan());
        let json = serde_json::to_string(&r).unwrap();
        let back: PanelResult = serde_json::from_str(&json).unwrap();
        assert!(back.f_stat.is_nan() && back.f_p_value.is_nan());
        assert_eq!(back.coefficients, r.coefficients);
    }
}
