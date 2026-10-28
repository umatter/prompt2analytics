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

use ndarray::{Array1, Array2, s};

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{eig_symmetric, safe_inverse, xtx, xty};
use crate::resource;
use crate::traits::estimator::{
    SignificanceLevel, chi_squared_p_value, f_test_p_value, normal_cdf, t_test_p_value,
};

use super::types::{
    ClusterAdjustment, PanelConfig, PanelMethod, PanelResult, PanelVcov, TestDistribution,
};
use super::utils::{demean_by_entity, demean_matrix_by_entity, extract_entity_ids};

/// Run Fixed Effects (within) panel estimation.
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
    // Extract y
    let y = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    // Build design matrix without intercept for FE
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?;

    let x = design.data;
    let var_names = design.column_names;
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
    let xtx_mat = xtx(&x_demeaned.view());
    let (xtx_inv, _cond_warning) =
        safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X in Fixed Effects".to_string(),
            suggestion: format!(
                "Check for perfect multicollinearity among variables: {:?}",
                e
            ),
        })?;

    let xty_vec = xty(&x_demeaned.view(), &y_demeaned);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Residuals from demeaned model
    let y_hat: Array1<f64> = x_demeaned.dot(&beta);
    let residuals = &y_demeaned - &y_hat;

    // Sum of squared residuals
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = ssr / df as f64;

    // Classical (homoskedastic) FE variance, matching R's
    // `plm(..., model = "within")` default: Var(β̂) = sigma^2 * (X̃'X̃)^{-1}.
    let vcov = &xtx_inv * sigma2;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v: f64| v.max(0.0).sqrt()).to_vec();

    // R-squared (within)
    let y_mean_demeaned = y_demeaned.mean().unwrap_or(0.0);
    let sst: f64 = y_demeaned
        .iter()
        .map(|y| (y - y_mean_demeaned).powi(2))
        .sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted R-squared
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - n_groups - 1) as f64) / (df as f64);

    // F-statistic
    let f_stat = if k > 0 && ssr > 0.0 {
        (sst - ssr) / (k as f64) / (ssr / df as f64)
    } else {
        0.0
    };
    let f_p_value = crate::traits::estimator::f_test_p_value(f_stat, k as f64, df as f64);

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
        vcov_type: PanelVcov::Classical,
        test_distribution: TestDistribution::StudentT,
        wald_chi2: None,
        wald_df: None,
        wald_p_value: None,
        theta_by_group: Vec::new(),
        warnings: Vec::new(),
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
///    design (K + 1 unless a regressor does not vary between entities). A
///    negative σ²_η is set to 0 with a warning, as plm.
/// 3. `θ_i = 1 - sqrt(σ²_ν / (T_i σ²_η + σ²_ν))`, quasi-demeaning
///    `y*_it = y_it - θ_i ȳ_i`, `X*_it = X_it - θ_i X̄_i`, and OLS on the
///    transformed model.
///
/// The covariance is `config.vcov`: classical `s² (X*'X*)⁻¹` with `s²` the
/// transformed residual variance (plm's `vcov`), or the entity-clustered
/// (Arellano) sandwich. Inference is normal (z), the joint test of the slopes is
/// a Wald chi-squared with its F form in `f_stat`, and R² is
/// `cor(y*, X*b)²`, all as plm's `summary`.
///
/// Refused: `n <= N + K` (no within degrees of freedom for σ²_ν) and
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

    // Design with intercept (column 0, "(Intercept)") and the slopes alone
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x = design.data;
    let var_names = design.column_names;
    let x_s = DesignMatrix::from_dataframe(dataset.df(), x_cols, false)?.data;
    let n = y.len();
    let p = x.ncols(); // K + 1
    let k_s = x_s.ncols(); // K

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
    let y_w = demean_by_entity(&y, &entity_ids, n_groups);
    let x_w = demean_matrix_by_entity(&x_s, &entity_ids, n_groups);
    let (xtx_w_inv, _) =
        safe_inverse(&xtx(&x_w.view()).view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (demeaned) in Random Effects".to_string(),
            suggestion: format!("Check for perfect multicollinearity: {:?}", e),
        })?;
    let b_w: Array1<f64> = xtx_w_inv.dot(&xty(&x_w.view(), &y_w));
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

    // Step 3: between regression, rank-aware (a regressor without between
    // variation makes A singular; plm then drops the aliased column)
    let (a_inv, rank_b) = between_inverse(&a)?;
    if rank_b < p {
        warnings.push(format!(
            "the Swamy-Arora between regression is rank deficient (rank {rank_b} of {p}): a \
             regressor does not vary between entities, so a generalized inverse was used (plm \
             drops the aliased regressor; on balanced panels the result is identical)"
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

    // Step 5: OLS on the transformed model
    let (bread, _) =
        safe_inverse(&xtx(&x_q.view()).view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (quasi-demeaned) in Random Effects".to_string(),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        })?;
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
    // its F form W/K (plm's pwaldtest(test = "F"))
    let b_s = beta.slice(s![1..]).to_owned();
    let v_ss = vcov.slice(s![1.., 1..]).to_owned();
    // safe_inverse returns a pseudoinverse with Some(condition) for a singular
    // matrix, never Err; only a clean inverse gives a Wald statistic.
    let (wald_chi2, wald_df, wald_p_value, f_stat, f_p_value) = match safe_inverse(&v_ss.view()) {
        Ok((v_inv, None)) => {
            let w = b_s.dot(&v_inv.dot(&b_s));
            let kf = k_s as f64;
            (
                Some(w),
                Some(k_s),
                Some(chi_squared_p_value(w, kf)),
                w / kf,
                f_test_p_value(w / kf, kf, df as f64),
            )
        }
        Err(_) | Ok((_, Some(_))) => {
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
/// The rank is judged on the unit-diagonal scaling `S A S`, `S = diag(A_jj^-1/2)`,
/// against `100 (K + 1) ε max(1, λ_max)`, so regressor units do not matter. At
/// full rank this is `safe_inverse(A)`; otherwise the generalized inverse
/// `S (Σ_{λ_j > tol} u_j u_j' / λ_j) S`. Any generalized inverse gives the same
/// Swamy-Arora `quad2` and trace term, because the range of `Σ T_i² z̄ z̄'` lies
/// in the range of `A`.
fn between_inverse(a: &Array2<f64>) -> EconResult<(Array2<f64>, usize)> {
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
    if rank == p {
        let (inv, _) = safe_inverse(&a.view()).map_err(|e| EconError::SingularMatrix {
            context: "X'X (between) in Random Effects".to_string(),
            suggestion: format!("Check for multicollinearity: {:?}", e),
        })?;
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
    let (n_clusters, p) = scores.dim();
    let mut meat = Array2::<f64>::zeros((p, p));
    for i in 0..n_clusters {
        resource::checkpoint()?;
        for a in 0..p {
            for b in 0..p {
                meat[[a, b]] += scores[[i, a]] * scores[[i, b]];
            }
        }
    }
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

    #[test]
    fn test_random_effects_stops_when_cancelled() {
        let ds = synthetic_panel(50, 4);
        let res = resource::with_budget(cancelled(), || {
            run_random_effects(&ds, "y", &["x"], "entity")
        });
        assert!(matches!(res, Err(EconError::Cancelled(_))), "{res:?}");
    }
}
