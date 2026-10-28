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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use crate::data::Dataset;
use crate::econometrics::hdfe::reject_non_finite;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignError, DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{eig_symmetric, safe_inverse, xtx, xty};
use crate::resource;
use crate::traits::estimator::{
    SignificanceLevel, chi_squared_p_value, f_test_p_value, normal_cdf, t_test_p_value,
};

use super::types::{
    ClusterAdjustment, PanelConfig, PanelMethod, PanelResult, PanelVcov, TestDistribution,
};
use super::utils::{demean_by_entity, demean_matrix_by_entity, extract_entity_ids};

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
///
/// Refused: a non-finite value in `y` or a regressor, and regressors that are
/// exactly collinear after removing the entity means (the coefficients are not
/// identified).
pub fn run_fixed_effects_with(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    config: &PanelConfig,
) -> EconResult<PanelResult> {
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|_| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Build design matrix without intercept for FE
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;

    let x = design.data;
    let var_names = design.column_names;
    reject_non_finite_data(&y, y_col, &x, &var_names)?;
    let n = y.len();
    let k = x.ncols();

    // Extract entity IDs
    let (entity_ids, n_groups) = extract_entity_ids(dataset, entity_col)?;

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "At least 2 entities required for panel estimation".to_string(),
        });
    }

    // Demean y and X by entity
    let y_demeaned = demean_by_entity(&y, &entity_ids, n_groups);
    let x_demeaned = demean_matrix_by_entity(&x, &entity_ids, n_groups);

    // Degrees of freedom: n - N - k (lost N entity dummies)
    let df = n.saturating_sub(n_groups).saturating_sub(k);
    if df == 0 {
        return Err(EconError::InsufficientData {
            required: n_groups + k + 1,
            provided: n,
            context: "Not enough observations for degrees of freedom".to_string(),
        });
    }

    // OLS on demeaned data: β = (X̃'X̃)^{-1} X̃'ỹ
    let mut warnings = Vec::new();
    let xtx_mat = xtx(&x_demeaned.view());
    require_full_rank(
        &xtx_mat,
        &var_names,
        "X'X in Fixed Effects",
        "after removing the entity means the regressors",
        " (a regressor that is constant within every entity is zero after demeaning)",
    )?;
    let (xtx_inv, cond) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "X'X in Fixed Effects".to_string(),
        suggestion: format!(
            "Check for perfect multicollinearity among variables: {:?}",
            e
        ),
    })?;
    warn_pseudoinverse(
        &mut warnings,
        "the within regression",
        "the coefficients may not be identified",
        cond,
    );

    let xty_vec = xty(&x_demeaned.view(), &y_demeaned);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

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
            let scores = entity_scores(&x_demeaned, &residuals, &entity_ids, n_groups)?;
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
    // classical within F, or the Wald F W/K on (K, G - 1) for a cluster vcov
    let (f_stat, f_p_value) = match config.vcov {
        PanelVcov::Classical => {
            let f = if k > 0 && ssr > 0.0 {
                (sst - ssr) / (k as f64) / (ssr / df as f64)
            } else {
                0.0
            };
            (f, f_test_p_value(f, k as f64, df as f64))
        }
        PanelVcov::Arellano(_) => match wald_statistic(&beta.view(), &vcov.view()) {
            Some(w) => {
                let f = w / k as f64;
                (f, f_test_p_value(f, k as f64, (n_groups - 1) as f64))
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
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
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
/// 1. σ²_ν from the within regression, `SSR_w / (n - N - K)`.
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
/// Refused: a non-finite value in `y` or a regressor, regressors that are
/// exactly collinear with each other or with the intercept (the coefficients are
/// not identified), `n <= N + K` (no within degrees of freedom for σ²_ν) and
/// `N <= K + 1` (the between regression is not estimable, as plm).
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
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|_| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Design with intercept (column 0, "(Intercept)"); the slopes are 1..
    if x_cols.is_empty() {
        return Err(DesignError::NoColumns.into());
    }
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = design.data;
    let var_names = design.column_names;
    reject_non_finite_data(&y, y_col, &x, &var_names)?;
    let n = y.len();
    let p = x.ncols(); // K + 1
    let k_s = p - 1; // K

    let (entity_ids, n_groups) = extract_entity_ids(dataset, entity_col)?;

    if n_groups < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_groups,
            context: "At least 2 entities required for panel estimation".to_string(),
        });
    }
    if n <= n_groups + k_s {
        return Err(EconError::InsufficientData {
            required: n_groups + k_s + 1,
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

    let mut warnings = Vec::new();

    // Step 1: within regression on the entity-demeaned slopes -> sigma^2_nu
    // (the demeaned intercept column is zero and is left out of the view)
    let y_w = demean_by_entity(&y, &entity_ids, n_groups);
    let x_demeaned = demean_matrix_by_entity(&x, &entity_ids, n_groups);
    let x_w = x_demeaned.slice(s![.., 1..]);
    let (xtx_w_inv, cond) =
        safe_inverse(&xtx(&x_w).view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (demeaned) in Random Effects".to_string(),
            suggestion: format!("Check for perfect multicollinearity: {:?}", e),
        })?;
    warn_pseudoinverse(
        &mut warnings,
        "the within regression that estimates sigma_e^2",
        "a regressor may not vary within entities, and sigma_e^2 still counts it in its \
         degrees of freedom",
        cond,
    );
    let b_w: Array1<f64> = xtx_w_inv.dot(&xty(&x_w, &y_w));
    let e_w = &y_w - &x_w.dot(&b_w);
    let sigma2_nu = e_w.dot(&e_w) / (n - n_groups - k_s) as f64;

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
    let (a_inv, rank_b) = between_inverse(&a, &mut warnings)?;
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
    // (average over t: (1 - theta_i) xbar_i'c = 0 with theta_i < 1), so its rank
    // is that of the data with the intercept.
    let xtx_q = xtx(&x_q.view());
    require_full_rank(
        &xtx_q,
        &var_names,
        "X'X (quasi-demeaned) in Random Effects",
        "the regressors and the intercept",
        "",
    )?;
    let (bread, cond) = safe_inverse(&xtx_q.view()).map_err(|e| EconError::SingularMatrix {
        context: "X'X (quasi-demeaned) in Random Effects".to_string(),
        suggestion: format!("Check for multicollinearity: {:?}", e),
    })?;
    warn_pseudoinverse(
        &mut warnings,
        "the quasi-demeaned regression",
        "the coefficients may not be identified",
        cond,
    );
    let beta: Array1<f64> = bread.dot(&xty(&x_q.view(), &y_q));
    let fitted = x_q.dot(&beta);
    let residuals = &y_q - &fitted;
    let df = n - p;
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
    let f_df2 = match config.vcov {
        PanelVcov::Classical => df,
        PanelVcov::Arellano(_) => n_groups - 1,
    };
    let wald = wald_statistic(&beta.slice(s![1..]), &vcov.slice(s![1.., 1..]));
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
        coef_arr,
        se_arr,
        residuals,
        vcov,
    })
}

/// Refuses a non-finite value in `y` or in a column of `x`, as `run_hdfe` does.
fn reject_non_finite_data(
    y: &Array1<f64>,
    y_col: &str,
    x: &Array2<f64>,
    names: &[String],
) -> EconResult<()> {
    reject_non_finite(y.iter(), &format!("Dependent variable '{y_col}'"))?;
    for (j, name) in names.iter().enumerate() {
        reject_non_finite(x.column(j).iter(), &format!("Regressor '{name}'"))?;
    }
    Ok(())
}

/// Records that `safe_inverse` fell back to a pseudoinverse (`Some(cond)`) for
/// the cross-product matrix of `what`, with its `consequence`.
fn warn_pseudoinverse(
    warnings: &mut Vec<String>,
    what: &str,
    consequence: &str,
    cond: Option<f64>,
) {
    if let Some(c) = cond {
        warnings.push(format!(
            "the cross-product matrix of {what} is ill-conditioned (condition number {c:.2e}), \
             so a pseudoinverse was used: {consequence}"
        ));
    }
}

/// `b' V⁻¹ b`, or `None` when `V` is singular. For a singular matrix
/// `safe_inverse` may return a pseudoinverse with `Some(condition)` instead of
/// `Err`; only a clean inverse gives a Wald statistic.
fn wald_statistic(b: &ArrayView1<f64>, v: &ArrayView2<f64>) -> Option<f64> {
    match safe_inverse(v) {
        Ok((v_inv, None)) => Some(b.dot(&v_inv.dot(b))),
        Err(_) | Ok((_, Some(_))) => None,
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

/// Refuses a cross-product `X'X` of less than full rank (judged as in
/// [`scaled_eigen`]): exactly collinear columns, or a column that is
/// identically zero, are not identified. The error names the columns in the
/// null space, `"{subject} are exactly collinear (rank r of p; involved: ...)"`,
/// followed by `hint`. A within-demeaned column of rounding residue (a regressor
/// that does not vary within entities, with entity means that are not exact in
/// floating point) is not caught here.
fn require_full_rank(
    xtx: &Array2<f64>,
    names: &[String],
    context: &str,
    subject: &str,
    hint: &str,
) -> EconResult<()> {
    let p = xtx.nrows();
    let e = scaled_eigen(xtx)?;
    if e.rank == p {
        return Ok(());
    }
    let mut involved = vec![false; p];
    for (j, &l) in e.lambda.iter().enumerate() {
        if l <= e.tol {
            for (i, flag) in involved.iter_mut().enumerate() {
                if e.s[i] == 0.0 || e.u[[i, j]].abs() > 1e-6 {
                    *flag = true;
                }
            }
        }
    }
    let shown: Vec<&str> = names
        .iter()
        .zip(&involved)
        .filter(|&(_, &f)| f)
        .map(|(n, _)| n.as_str())
        .collect();
    Err(EconError::SingularMatrix {
        context: context.to_string(),
        suggestion: format!(
            "{subject} are exactly collinear (rank {} of {p}; involved: {}), so their \
             coefficients are not identified: drop one of them{hint}",
            e.rank,
            shown.join(", ")
        ),
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
/// `100 (K + 1) ε max(1, λ_max)`. At full rank this is `safe_inverse(A)` (a
/// pseudoinverse fallback is recorded in `warnings`); otherwise the generalized
/// inverse `S (Σ_{λ_j > tol} u_j u_j' / λ_j) S`. Any generalized inverse gives
/// the same Swamy-Arora `quad2` and trace term, because the range of
/// `Σ T_i² z̄ z̄'` lies in the range of `A`.
fn between_inverse(
    a: &Array2<f64>,
    warnings: &mut Vec<String>,
) -> EconResult<(Array2<f64>, usize)> {
    let p = a.nrows();
    let ScaledEigen {
        s,
        lambda,
        u,
        tol,
        rank,
    } = scaled_eigen(a)?;
    if rank == p {
        let (inv, cond) = safe_inverse(&a.view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (between) in Random Effects".to_string(),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        })?;
        warn_pseudoinverse(
            warnings,
            "the between regression",
            "the variance components may be inaccurate",
            cond,
        );
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

    /// The message of an `InvalidSpecification` error, or a panic naming `res`.
    fn invalid_message(res: EconResult<PanelResult>) -> String {
        match res {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// A NaN in y used to give NaN coefficients with SE 0 and p 1; it is
    /// refused as in `run_hdfe`, as is an infinite regressor value.
    #[test]
    fn test_fixed_effects_refuses_non_finite_data() {
        let fe = |ds: &Dataset| run_fixed_effects(ds, "y", &["x"], "entity");
        let m = invalid_message(fe(&panel_with("y", 5, f64::NAN)));
        assert!(
            m.contains("Dependent variable 'y'") && m.contains("1 non-finite"),
            "{m}"
        );
        let m = invalid_message(fe(&panel_with("x", 7, f64::INFINITY)));
        assert!(m.contains("Regressor 'x'"), "{m}");
        let config = PanelConfig {
            vcov: PanelVcov::Arellano(ClusterAdjustment::Stata),
        };
        let ds = panel_with("y", 0, f64::NAN);
        invalid_message(run_fixed_effects_with(&ds, "y", &["x"], "entity", &config));
    }

    /// RE with a NaN in y used to return NaN coefficients, SE 0, p 1 and a
    /// spurious between-rank warning.
    #[test]
    fn test_random_effects_refuses_non_finite_data() {
        let re = |ds: &Dataset| run_random_effects(ds, "y", &["x"], "entity");
        let m = invalid_message(re(&panel_with("y", 5, f64::NAN)));
        assert!(
            m.contains("Dependent variable 'y'") && m.contains("1 non-finite"),
            "{m}"
        );
        let m = invalid_message(re(&panel_with("x", 7, f64::NEG_INFINITY)));
        assert!(m.contains("Regressor 'x'"), "{m}");
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
}
