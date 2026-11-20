//! Spatial diagnostics and tests.
//!
//! Provides tests for spatial autocorrelation (Moran's I, Geary's C)
//! and Lagrange Multiplier tests for spatial dependence in regression residuals.
//!
//! Moran's I and Geary's C follow spdep's `zero.policy = TRUE, adjust.n = TRUE`: a unit without
//! neighbours stays in the mean and the kurtosis, but n in the statistic and its moments is
//! n - #islands. Both tests need four units with neighbours, and refuse a W without links and
//! constant or non-finite data, where spdep returns NaN. A permutation variance that is not
//! positive is an error. See `validation/spatial/moran.md`; the residual Moran moments, the LM
//! tests and `localmoran` are not pinned to R.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::traits::estimator::{chi_squared_p_value, normal_p_value_two_sided};

use super::weights::SpatialWeights;
use crate::errors::{EconError, EconResult};

/// Alternative hypothesis for Moran's I test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MoranAlternative {
    /// Two-sided test (I ≠ E[I])
    #[default]
    TwoSided,
    /// Greater than expected (positive autocorrelation)
    Greater,
    /// Less than expected (negative autocorrelation)
    Less,
}

crate::option_enum!(
    MoranAlternative,
    crate::options::Choices {
        option: "alternative",
        values: &[
            (
                &["two.sided", "twosided", "both"],
                MoranAlternative::TwoSided
            ),
            (&["greater", "positive"], MoranAlternative::Greater),
            (&["less", "negative"], MoranAlternative::Less),
        ],
    }
);

/// Result of Moran's I test for spatial autocorrelation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoranResult {
    /// Moran's I statistic
    pub statistic: f64,
    /// Expected value under null hypothesis
    pub expectation: f64,
    /// Variance under null hypothesis
    pub variance: f64,
    /// Standardized z-score
    pub z_score: f64,
    /// p-value
    pub p_value: f64,
    /// Alternative hypothesis used
    pub alternative: MoranAlternative,
    /// Number of observations
    pub n: usize,
    /// Units without neighbours. `moran_test` and `geary_test` use n - n_islands in the
    /// statistic and its moments (spdep `adjust.n = TRUE`); `moran_test_residuals` uses the
    /// linked units (spdep `lm.morantest(zero.policy = TRUE)`). `n` stays the row count.
    #[serde(default)]
    pub n_islands: usize,
}

/// Result of Geary's C test for spatial autocorrelation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GearyResult {
    /// Geary's C statistic
    pub statistic: f64,
    /// Expected value under null hypothesis (= 1)
    pub expectation: f64,
    /// Variance under null hypothesis
    pub variance: f64,
    /// Standardized z-score
    pub z_score: f64,
    /// p-value (two-sided)
    pub p_value: f64,
    /// Number of observations
    pub n: usize,
    /// Units without neighbours; the statistic uses n - n_islands (spdep adjust.n = TRUE)
    #[serde(default)]
    pub n_islands: usize,
}

/// Result of a single LM test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LmTestResult {
    /// Test statistic
    pub statistic: f64,
    /// Degrees of freedom
    pub df: usize,
    /// p-value
    pub p_value: f64,
}

/// LISA cluster classification.
///
/// Classifies each observation based on its value relative to the mean
/// and its neighbors' values relative to the mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LisaCluster {
    /// High value surrounded by high values (hot spot)
    HighHigh,
    /// Low value surrounded by low values (cold spot)
    LowLow,
    /// High value surrounded by low values (spatial outlier)
    HighLow,
    /// Low value surrounded by high values (spatial outlier)
    LowHigh,
    /// Not significant
    NotSignificant,
}

/// Result for a single observation in Local Moran's I analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalMoranObs {
    /// Local Moran's I statistic for this observation
    pub i_local: f64,
    /// Expected value under null
    pub expectation: f64,
    /// Variance under null
    pub variance: f64,
    /// Z-score
    pub z_score: f64,
    /// P-value
    pub p_value: f64,
    /// LISA cluster classification (if significant)
    pub cluster: LisaCluster,
}

/// Result of Local Moran's I (LISA) analysis.
///
/// Local Indicators of Spatial Association decompose global Moran's I
/// into contributions from each observation, allowing identification
/// of local clusters and spatial outliers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalMoranResult {
    /// Local statistics for each observation
    pub local_stats: Vec<LocalMoranObs>,
    /// Global Moran's I, the sum of the local I over S0, with m2 over all n units (spdep
    /// `adjust.n = FALSE`); with islands, n / (n - islands) times [`moran_test`]'s statistic
    pub global_i: f64,
    /// Number of observations
    pub n: usize,
    /// Significance level used for cluster classification
    pub alpha: f64,
    /// Count of High-High clusters
    pub n_high_high: usize,
    /// Count of Low-Low clusters
    pub n_low_low: usize,
    /// Count of High-Low outliers
    pub n_high_low: usize,
    /// Count of Low-High outliers
    pub n_low_high: usize,
    /// Count of not significant
    pub n_not_sig: usize,
    /// Whether the moments are under conditional randomisation (spdep's default) or total
    #[serde(default)]
    pub conditional: bool,
}

/// Rao score (Lagrange multiplier) tests for spatial dependence in a linear regression
/// (spdep `lm.RStests`; spdep's names in brackets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialLmTests {
    /// Score test for a spatial lag, H0: ρ = 0 (RSlag)
    pub lm_lag: LmTestResult,
    /// Score test for a spatial error, H0: λ = 0 (RSerr)
    pub lm_error: LmTestResult,
    /// Lag test robust to a local spatial error (adjRSlag)
    pub rlm_lag: LmTestResult,
    /// Error test robust to a local spatial lag (adjRSerr)
    pub rlm_error: LmTestResult,
    /// Joint test, H0: ρ = λ = 0 (SARMA, df 2)
    pub lm_sarma: LmTestResult,
    /// Warnings (spdep's "Spatial weights matrix not row standardized")
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// What the global tests share: the moments of the centred variable and spdep's weight constants.
struct WeightConstants {
    /// Σ z_i²
    zz: f64,
    /// Σ z_i⁴
    z4: f64,
    s0: f64,
    /// Σ_ij term(i, j, w_ij), summed in the one pass over the links that S0 needs
    numerator: f64,
    /// n - n_islands
    n_adj: usize,
    n_islands: usize,
}

/// spdep spweights.constants(zero.policy = TRUE, adjust.n = TRUE) plus the checks spdep leaves to
/// NaN: finite and non-constant x, at least one link, n_adj = n - #islands >= 4.
fn weight_constants(
    x: &Array1<f64>,
    listw: &SpatialWeights,
    what: &str,
    term: impl Fn(&Array1<f64>, usize, usize, f64) -> f64,
) -> EconResult<WeightConstants> {
    let n = x.len();
    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: format!("{what}: variable length must match weights matrix size"),
        });
    }
    let bad = x.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        return Err(EconError::InvalidSpecification {
            message: format!("{what}: x contains {bad} non-finite value(s)"),
        });
    }
    if n > 0 && x.iter().all(|&v| v == x[0]) {
        return Err(EconError::InvalidSpecification {
            message: format!("{what}: x is constant, so spatial autocorrelation is undefined"),
        });
    }
    let mean = x.sum() / n as f64;
    let z: Array1<f64> = x.mapv(|v| v - mean);
    let mut s0 = 0.0;
    let mut numerator = 0.0;
    let mut n_islands = 0;
    for i in 0..n {
        let sw = listw.get_weights(i);
        if sw.indices.is_empty() {
            n_islands += 1;
        }
        for (&j, &w_ij) in sw.indices.iter().zip(sw.weights.iter()) {
            s0 += w_ij;
            numerator += term(&z, i, j, w_ij);
        }
    }
    if !(s0 > 0.0 && s0.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "{what}: the spatial weights have no links (S0 = 0): every unit is an island; \
                 use a larger distance band or more neighbours"
            ),
        });
    }
    let n_adj = n - n_islands;
    if n_adj < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n_adj,
            context: format!(
                "{what}: units with at least one neighbour (spdep adjust.n = TRUE); the \
                 randomisation variance divides by (n - 1)(n - 2)(n - 3)"
            ),
        });
    }
    let (mut zz, mut z4) = (0.0, 0.0);
    for &v in z.iter() {
        let v2 = v * v;
        zz += v2;
        z4 += v2 * v2;
    }
    if !(zz.is_finite() && z4.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("{what}: the moments of x overflow; rescale x"),
        });
    }
    Ok(WeightConstants {
        zz,
        z4,
        s0,
        numerator,
        n_adj,
        n_islands,
    })
}

/// Moran's I test for spatial autocorrelation.
///
/// Tests whether the values of a variable are spatially correlated.
/// Positive autocorrelation means similar values cluster together,
/// negative means dissimilar values are neighbors.
///
/// # Arguments
///
/// * `x` - Variable to test
/// * `listw` - Spatial weights
/// * `alternative` - Alternative hypothesis
///
/// # Returns
///
/// Test result with statistic, p-value, etc.
///
/// # Errors
///
/// - `InvalidSpecification`: `x` and the weights differ in length, `x` has a non-finite value
///   or is constant, its moments overflow, or the weights have no links (S0 = 0).
/// - `InsufficientData`: fewer than 4 units with a neighbour.
/// - `Computation`: the randomisation variance is not positive (a degenerate permutation
///   distribution of `x`).
///
/// # Example
///
/// ```
/// use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle, moran_test, MoranAlternative};
/// use ndarray::array;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
/// let nb = Neighbors::from_knn(&coords, 2);
/// let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
/// let y = array![1.0, 2.0, 1.5, 2.5];
///
/// let result = moran_test(&y, &listw, MoranAlternative::TwoSided)?;
/// println!("Moran's I = {}, p-value = {}", result.statistic, result.p_value);
/// # Ok(())
/// # }
/// ```
pub fn moran_test(
    x: &Array1<f64>,
    listw: &SpatialWeights,
    alternative: MoranAlternative,
) -> EconResult<MoranResult> {
    let n = x.len();
    let WeightConstants {
        zz,
        z4,
        s0,
        numerator,
        n_adj,
        n_islands,
    } = weight_constants(x, listw, "moran_test", |z, i, j, w| w * z[i] * z[j])?;
    let nn = n_adj as f64;

    // Moran's I and its expectation under no spatial correlation
    let i_stat = (nn / s0) * (numerator / zz);
    let ei = -1.0 / (nn - 1.0);

    // Variance under randomisation (Cliff & Ord 1981); the kurtosis is over all n, as spdep's
    // moran() (length(x))
    let (s1, s2) = compute_s1_s2(listw);
    let s0_sq = s0 * s0;
    let b2 = n as f64 * z4 / (zz * zz);
    let term1 = nn * ((nn * nn - 3.0 * nn + 3.0) * s1 - nn * s2 + 3.0 * s0_sq);
    let term2 = b2 * ((nn * nn - nn) * s1 - 2.0 * nn * s2 + 6.0 * s0_sq);
    let term3 = (nn - 1.0) * (nn - 2.0) * (nn - 3.0) * s0_sq;
    let vi = (term1 - term2) / term3 - ei * ei;
    if vi.is_nan() || vi <= 1e-8 * ei * ei {
        return Err(EconError::Computation(format!(
            "Moran's I randomisation variance is not positive ({vi:e}): the permutation \
             distribution of x is degenerate (spdep warns \"Negative variance\")"
        )));
    }

    // Z-score and p-value
    let z_score = (i_stat - ei) / vi.sqrt();
    let p_value = normal_p_value(z_score, alternative);

    Ok(MoranResult {
        statistic: i_stat,
        expectation: ei,
        variance: vi,
        z_score,
        p_value,
        alternative,
        n,
        n_islands,
    })
}

/// The p-value of a standard normal z under `alternative`.
fn normal_p_value(z: f64, alternative: MoranAlternative) -> f64 {
    let normal = Normal::standard();
    match alternative {
        MoranAlternative::TwoSided => normal_p_value_two_sided(z),
        MoranAlternative::Greater => normal.sf(z),
        MoranAlternative::Less => normal.cdf(z),
    }
}

/// Moran's I test for regression residuals, as spdep's `lm.morantest(zero.policy = TRUE)`.
///
/// Tests for spatial autocorrelation in OLS residuals, which indicates potential spatial
/// dependence not captured by the model. W enters through its symmetric part
/// U = (W + W')/2 (spdep `listw2U`), which leaves u'Wu, S0 and S1 unchanged; the moments
/// account for the estimated coefficients (Cliff & Ord 1981, under normal errors):
///
/// ```text
/// I = N/S0 u'Wu / u'u,   Z = U X,   C1 = X'Z,   C3 = (X'X)^-1 C1,   tr A = tr C3
/// E[I] = -N tr A / ((N - p) S0)
/// Var[I] = N^2 / (S0^2 (N - p)(N - p + 2)) (S1 + 2 tr C3^2 - 4 tr((X'X)^-1 Z'Z) - 2 (tr A)^2 / (N - p))
/// ```
///
/// N is the number of units with at least one neighbour (spdep `zero.policy = TRUE`), p the
/// rank of X: aliased columns are dropped as `lm` drops them (pivoted QR, tolerance 1e-7).
/// spdep's default alternative is [`MoranAlternative::Greater`].
///
/// # Arguments
///
/// * `residuals` - OLS residuals of y on `x`
/// * `x` - Design matrix (including the intercept column if the model has one)
/// * `listw` - Spatial weights
/// * `alternative` - Alternative hypothesis
///
/// # Errors
///
/// - `InvalidSpecification`: lengths differ, the weights have no links, the residuals are all
///   zero or not finite, or `x` has a non-finite entry.
/// - `InsufficientData`: N - p < 1.
/// - `Computation`: the variance is not positive (a degenerate design).
pub fn moran_test_residuals(
    residuals: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
    alternative: MoranAlternative,
) -> EconResult<MoranResult> {
    use crate::linalg::solve::{LM_QR_TOL, Qr};
    use ndarray::Axis;

    let n = residuals.len();
    if n != listw.n() || n != x.nrows() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "moran_test_residuals: the residuals ({n}), the rows of x ({}) and the weights \
                 ({}) must have the same length",
                x.nrows(),
                listw.n()
            ),
        });
    }

    // u'Wu = u'Uu, S0 and the islands in one pass over the links
    let mut numerator = 0.0;
    let mut s0 = 0.0;
    let mut n_islands = 0;
    for i in 0..n {
        let sw = listw.get_weights(i);
        if sw.indices.is_empty() {
            n_islands += 1;
        }
        for (&j, &w_ij) in sw.indices.iter().zip(sw.weights.iter()) {
            numerator += w_ij * residuals[i] * residuals[j];
            s0 += w_ij;
        }
    }
    if !(s0 > 0.0 && s0.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "moran_test_residuals: the spatial weights have no links (S0 = 0): every \
                      unit is an island; use a larger distance band or more neighbours"
                .to_string(),
        });
    }
    let denominator: f64 = residuals.iter().map(|&r| r * r).sum();
    if !(denominator > 0.0 && denominator.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "moran_test_residuals: the residuals are all zero or not finite, so spatial \
                      autocorrelation is undefined"
                .to_string(),
        });
    }

    // The kept columns of X (lm drops aliased ones) and (X'X)^-1 on them
    let qr = Qr::decompose(x.view(), LM_QR_TOL)?;
    let p = qr.rank();
    let n_linked = n - n_islands;
    let df =
        n_linked
            .checked_sub(p)
            .filter(|&d| d >= 1)
            .ok_or_else(|| EconError::InsufficientData {
                required: p + 1,
                provided: n_linked,
                context:
                    "moran_test_residuals: units with a neighbour minus the rank of x must be at \
                      least 1"
                        .to_string(),
            })?;
    let aliased = qr.aliased();
    let keep: Vec<usize> = (0..x.ncols()).filter(|j| !aliased.contains(j)).collect();
    let cov = qr.unscaled_covariance()?;
    let xtx_inv = cov.select(Axis(0), &keep).select(Axis(1), &keep);
    let xk = x.select(Axis(1), &keep);

    // Z = U X = (W X + W'X) / 2
    let mut z = crate::resource::try_zeros2(n, p, "moran_test_residuals: U X")?;
    for (c, col) in xk.columns().into_iter().enumerate() {
        let col = col.to_owned();
        let zc = 0.5 * (&listw.lag(&col) + &listw.lag_transpose(&col));
        z.column_mut(c).assign(&zc);
    }
    let c1 = xk.t().dot(&z);
    let c2 = z.t().dot(&z);
    let c3 = xtx_inv.dot(&c1);
    let tr_a: f64 = c3.diag().sum();
    let tr_a2: f64 = (0..p)
        .flat_map(|i| (0..p).map(move |j| (i, j)))
        .map(|(i, j)| c3[[i, j]] * c3[[j, i]])
        .sum();
    let tr_b: f64 = 4.0
        * (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| xtx_inv[[i, j]] * c2[[j, i]])
            .sum::<f64>();
    let (s1, _s2) = compute_s1_s2(listw);

    let nn = n_linked as f64;
    let dfp = df as f64;
    let i_stat = (nn / s0) * (numerator / denominator);
    let ei = -(nn * tr_a) / (dfp * s0);
    let vi = (nn * nn) / (s0 * s0 * dfp * (dfp + 2.0))
        * (s1 + 2.0 * tr_a2 - tr_b - 2.0 * tr_a * tr_a / dfp);
    if !(vi > 0.0 && vi.is_finite()) {
        return Err(EconError::Computation(format!(
            "moran_test_residuals: the variance of I is not positive ({vi:e}); the design leaves \
             no spatial variation in the residuals"
        )));
    }

    let z_score = (i_stat - ei) / vi.sqrt();
    Ok(MoranResult {
        statistic: i_stat,
        expectation: ei,
        variance: vi,
        z_score,
        p_value: normal_p_value(z_score, alternative),
        alternative,
        n,
        n_islands,
    })
}

/// Geary's C test for spatial autocorrelation.
///
/// An alternative to Moran's I that focuses on differences between neighbors.
/// C < 1 indicates positive autocorrelation, C > 1 indicates negative.
///
/// # Arguments
///
/// * `x` - Variable to test
/// * `listw` - Spatial weights
///
/// # Returns
///
/// Test result with statistic, p-value, etc.
///
/// # Errors
///
/// - `InvalidSpecification`: `x` and the weights differ in length, `x` has a non-finite value
///   or is constant, its moments overflow, or the weights have no links (S0 = 0).
/// - `InsufficientData`: fewer than 4 units with a neighbour.
/// - `Computation`: the randomisation variance is not above 1e-8 S1/S0² (its scale): a
///   degenerate permutation distribution of `x`.
pub fn geary_test(x: &Array1<f64>, listw: &SpatialWeights) -> EconResult<GearyResult> {
    let n = x.len();
    let WeightConstants {
        zz,
        z4,
        s0,
        numerator,
        n_adj,
        n_islands,
    } = weight_constants(x, listw, "geary_test", |z, i, j, w| {
        let diff = z[i] - z[j];
        w * diff * diff
    })?;
    let nn = n_adj as f64;

    // Geary's C (spdep geary(): n1 = n_adj - 1) and its expectation
    let c_stat = ((nn - 1.0) / (2.0 * s0)) * (numerator / zz);
    let ec = 1.0;

    // Variance under randomisation (Cliff & Ord 1981), the kurtosis over all n
    let (s1, s2) = compute_s1_s2(listw);
    let s0_sq = s0 * s0;
    let b2 = n as f64 * z4 / (zz * zz);
    let vc = ((nn - 1.0) * s1 * (nn * nn - 3.0 * nn + 3.0 - (nn - 1.0) * b2)
        - 0.25 * (nn - 1.0) * s2 * (nn * nn + 3.0 * nn - 6.0 - b2 * (nn * nn - nn + 2.0))
        + s0_sq * (nn * nn - 3.0 - b2 * (nn - 1.0) * (nn - 1.0)))
        / (nn * (nn - 2.0) * (nn - 3.0) * s0_sq);
    if geary_variance_degenerate(vc, s0, s1) {
        return Err(EconError::Computation(format!(
            "Geary's C randomisation variance is not positive ({vc:e}): the permutation \
             distribution of x is degenerate"
        )));
    }

    // Z-score and p-value (two-sided)
    let z_score = (c_stat - ec) / vc.sqrt();
    let p_value = normal_p_value_two_sided(z_score);

    Ok(GearyResult {
        statistic: c_stat,
        expectation: ec,
        variance: vc,
        z_score,
        p_value,
        n,
        n_islands,
    })
}

/// Options of [`localmoran_with`].
#[derive(Debug, Clone, PartialEq)]
pub struct LocalMoranConfig {
    /// Significance level for the cluster classification (default 0.05)
    pub alpha: f64,
    /// Conditional permutations for the p-values; 0 (default) uses the normal approximation
    pub permutations: usize,
    /// Moments under conditional randomisation (spdep's default, `true`) or total
    /// randomisation (`false`)
    pub conditional: bool,
    /// Seed of the permutations (default 42)
    pub seed: u64,
}

impl Default for LocalMoranConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            permutations: 0,
            conditional: true,
            seed: 42,
        }
    }
}

/// Whether Geary's randomisation variance is degenerate: NaN, or not above 1e-8 of the scale
/// S1/S0^2 that Var(C) has (about 2/(n k) for a k-regular binary W).
fn geary_variance_degenerate(vc: f64, s0: f64, s1: f64) -> bool {
    let floor = 1e-8 * s1 / (s0 * s0);
    // NaN on either side counts as degenerate
    vc.partial_cmp(&floor) != Some(std::cmp::Ordering::Greater)
}

/// Local Moran's I (LISA) with the default options of [`LocalMoranConfig`] (conditional
/// moments, seed 42) and the given `alpha` and `permutations`; see [`localmoran_with`].
///
/// # Example
///
/// ```
/// use p2a_core::spatial::{Neighbors, SpatialWeights, WeightStyle, localmoran, LisaCluster};
/// use ndarray::array;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
/// let nb = Neighbors::from_knn(&coords, 2);
/// let listw = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
/// let y = array![1.0, 2.0, 1.5, 2.5];
///
/// let result = localmoran(&y, &listw, 0.05, 0)?;
/// for (i, obs) in result.local_stats.iter().enumerate() {
///     if obs.cluster != LisaCluster::NotSignificant {
///         println!("Location {}: {:?}, p={:.4}", i, obs.cluster, obs.p_value);
///     }
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// As [`localmoran_with`].
pub fn localmoran(
    x: &Array1<f64>,
    listw: &SpatialWeights,
    alpha: f64,
    permutations: usize,
) -> EconResult<LocalMoranResult> {
    localmoran_with(
        x,
        listw,
        &LocalMoranConfig {
            alpha,
            permutations,
            ..LocalMoranConfig::default()
        },
    )
}

/// Local Moran's I (LISA) - Local Indicators of Spatial Association, as spdep's
/// `localmoran(mlvar = TRUE, alternative = "two.sided")`.
///
/// I_i = (z_i / m2) Σ_j w_ij z_j with z = x - mean(x) and m2 = Σ z² / n. The moments follow
/// spdep: under conditional randomisation (`conditional = true`, spdep's default), with
/// W_i = Σ_j w_ij and W_i2 = Σ_j w_ij²,
///
/// ```text
/// E[I_i] = -z_i² W_i / ((n - 1) m2)
/// Var[I_i] = (z_i / m2)² n/(n - 2) (W_i2 - W_i²/(n - 1)) (m2 - z_i²/(n - 1))
/// ```
///
/// and under total randomisation (`conditional = false`) E[I_i] = -W_i / (n - 1) with the
/// Cliff-Ord variance through the kurtosis b2. The p-value is the two-sided normal one, or
/// with `permutations > 0` the folded permutation p-value (spdep `localmoran_perm`'s
/// "Pr(folded) Sim"), (min(r, nsim - r) + 1) / (nsim + 1) with r the permutations whose I_i is
/// at least the observed one. Each permutation draws unit i's neighbour values without
/// repeats from the other n - 1 values (Anselin's conditional permutation; spdep's
/// `no_repeat_in_row = TRUE`, not its default). A zero variance (an island, or z_i = 0 under
/// conditional moments) gives a NaN z-score and p-value and [`LisaCluster::NotSignificant`],
/// where spdep gives NaN. `global_i` is Moran's I, Σ I_i / S0, with m2 over all n units
/// (spdep `adjust.n = FALSE`): when W has islands it is n / (n - islands) times the statistic
/// of [`moran_test`], which divides by the n - islands units with neighbours.
///
/// # Errors
///
/// - `InvalidSpecification`: `x` and the weights differ in length, `x` has a non-finite value
///   or is constant, fewer than 3 units, `alpha` is not in [0, 1], or a unit lists itself as
///   a neighbour (the moments and the permutation assume a zero diagonal).
/// - `Cancelled`: the installed budget expired during the permutations.
///
/// # Reference
///
/// Anselin, L. (1995). Local Indicators of Spatial Association—LISA.
/// Geographical Analysis, 27(2), 93-115.
pub fn localmoran_with(
    x: &Array1<f64>,
    listw: &SpatialWeights,
    config: &LocalMoranConfig,
) -> EconResult<LocalMoranResult> {
    let n = x.len();
    let alpha = config.alpha;
    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "localmoran: x has {n} values but the weights have {} units",
                listw.n()
            ),
        });
    }
    if n < 3 {
        return Err(EconError::InvalidSpecification {
            message: format!("localmoran: needs at least 3 observations, got {n}"),
        });
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(EconError::InvalidSpecification {
            message: format!("localmoran: alpha must lie in [0, 1], got {alpha}"),
        });
    }
    let bad = x.iter().filter(|v| !v.is_finite()).count();
    if bad > 0 {
        return Err(EconError::InvalidSpecification {
            message: format!("localmoran: x contains {bad} non-finite value(s)"),
        });
    }
    if x.iter().all(|&v| v == x[0]) {
        return Err(EconError::InvalidSpecification {
            message: "localmoran: x is constant, so local autocorrelation is undefined".to_string(),
        });
    }
    if let Some(i) = super::first_self_link(listw) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "localmoran: unit {} lists itself as a neighbour; W must have a zero diagonal",
                i + 1
            ),
        });
    }
    if config.permutations > 0
        && let Some(i) = (0..n).find(|&i| listw.get_weights(i).indices.len() > n - 1)
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "localmoran: unit {} has {} neighbours, more than the n - 1 = {} other units",
                i + 1,
                listw.get_weights(i).indices.len(),
                n - 1
            ),
        });
    }

    let n_f = n as f64;
    let mean = x.sum() / n_f;
    let z: Array1<f64> = x.mapv(|v| v - mean);
    let m2: f64 = z.iter().map(|&zi| zi * zi).sum::<f64>() / n_f;
    let b2 = z.iter().map(|&zi| zi.powi(4)).sum::<f64>() / n_f / (m2 * m2);
    if !(m2.is_finite() && b2.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "localmoran: the moments of x overflow; rescale x".to_string(),
        });
    }
    let wz = listw.lag(&z);
    let observed: Vec<f64> = (0..n).map(|i| (z[i] / m2) * wz[i]).collect();

    let perm_pvalues = if config.permutations > 0 {
        let nsim = config.permutations;
        let count = local_moran_permutation_counts(&z, m2, &observed, listw, nsim, config.seed)?;
        Some(
            (0..n)
                .map(|i| {
                    if listw.get_weights(i).indices.is_empty() || z[i] == 0.0 {
                        return f64::NAN;
                    }
                    let r = count[i].min(nsim - count[i]);
                    (r as f64 + 1.0) / (nsim as f64 + 1.0)
                })
                .collect::<Vec<f64>>(),
        )
    } else {
        None
    };

    let mut local_stats = Vec::with_capacity(n);
    for i in 0..n {
        let (z_i, wz_i, i_local) = (z[i], wz[i], observed[i]);
        let sw = listw.get_weights(i);
        let w_i: f64 = sw.weights.iter().sum();
        let w_i2: f64 = sw.weights.iter().map(|&w| w * w).sum();
        let (ei, vi) = if config.conditional {
            let ei = -(z_i * z_i * w_i) / ((n_f - 1.0) * m2);
            let vi = (z_i / m2).powi(2)
                * (n_f / (n_f - 2.0))
                * (w_i2 - w_i * w_i / (n_f - 1.0))
                * (m2 - z_i * z_i / (n_f - 1.0));
            (ei, vi)
        } else {
            let ei = -w_i / (n_f - 1.0);
            let a = (n_f - b2) / (n_f - 1.0);
            let b = (2.0 * b2 - n_f) / ((n_f - 1.0) * (n_f - 2.0));
            (ei, a * w_i2 + b * (w_i * w_i - w_i2) - ei * ei)
        };
        // z = (I - E) / sqrt(V); NaN for a zero variance, as spdep
        let z_score = if vi > 0.0 {
            (i_local - ei) / vi.sqrt()
        } else {
            f64::NAN
        };
        let p_value = if z_score.is_nan() {
            f64::NAN
        } else if let Some(ref pvals) = perm_pvalues {
            pvals[i]
        } else {
            // once per unit: the erfc lower tail 2 Phi(-|z|), accurate far out and several
            // times cheaper than the chi-square route of `normal_p_value_two_sided`
            2.0 * Normal::standard().cdf(-z_score.abs())
        };

        let cluster = if p_value <= alpha {
            if z_i > 0.0 && wz_i > 0.0 {
                LisaCluster::HighHigh
            } else if z_i < 0.0 && wz_i < 0.0 {
                LisaCluster::LowLow
            } else if z_i > 0.0 && wz_i < 0.0 {
                LisaCluster::HighLow
            } else {
                LisaCluster::LowHigh
            }
        } else {
            LisaCluster::NotSignificant
        };

        local_stats.push(LocalMoranObs {
            i_local,
            expectation: ei,
            variance: vi,
            z_score,
            p_value,
            cluster,
        });
    }

    let count = |c: LisaCluster| local_stats.iter().filter(|o| o.cluster == c).count();
    let (n_high_high, n_low_low, n_high_low, n_low_high, n_not_sig) = (
        count(LisaCluster::HighHigh),
        count(LisaCluster::LowLow),
        count(LisaCluster::HighLow),
        count(LisaCluster::LowHigh),
        count(LisaCluster::NotSignificant),
    );
    Ok(LocalMoranResult {
        global_i: observed.iter().sum::<f64>() / listw.sum_weights(),
        local_stats,
        n,
        alpha,
        n_high_high,
        n_low_low,
        n_high_low,
        n_low_high,
        n_not_sig,
        conditional: config.conditional,
    })
}

/// Conditional-permutation counts of local Moran's I (spdep `localmoran_perm` with
/// `no_repeat_in_row = TRUE`): in each of `nsim` permutations unit i's neighbours receive
/// |N_i| values drawn without repeats from the other n - 1 values, and r_i counts the
/// permutations with I_perm >= I_i. The caller folds them into
/// p_i = (min(r_i, nsim - r_i) + 1) / (nsim + 1) ("Pr(folded) Sim"). The draws of
/// permutation r come from a generator keyed by (seed, r). The weights must have a zero
/// diagonal ([`localmoran_with`] refuses a self-link), so no row has more than n - 1 entries.
fn local_moran_permutation_counts(
    z: &Array1<f64>,
    m2: f64,
    observed: &[f64],
    listw: &SpatialWeights,
    nsim: usize,
    seed: u64,
) -> EconResult<Vec<usize>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let n = z.len();
    let mut count = vec![0usize; n];
    // A permutation of 0..n-1 (offsets among the other n - 1 units): a partial Fisher-Yates
    // shuffle of any arrangement draws a uniform sample without repeats.
    let mut pool: Vec<usize> = (0..n - 1).collect();
    // I_perm >= I_obs up to rounding in the last digits, so that the observed neighbour set
    // itself counts
    let threshold: Vec<f64> = observed.iter().map(|&v| v - 1e-12 * v.abs()).collect();
    for rep in 0..nsim {
        crate::resource::checkpoint()?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(rep as u64);
        for i in 0..n {
            let sw = listw.get_weights(i);
            if sw.indices.is_empty() {
                continue;
            }
            let mut lag = 0.0;
            for (t, &w) in sw.weights.iter().enumerate() {
                let r = rng.gen_range(t..n - 1);
                pool.swap(t, r);
                let off = pool[t];
                let j = if off >= i { off + 1 } else { off };
                lag += w * z[j];
            }
            if (z[i] / m2) * lag >= threshold[i] {
                count[i] += 1;
            }
        }
    }
    Ok(count)
}

/// Rao score (Lagrange multiplier) tests for spatial dependence in a linear regression, as
/// spdep's `lm.RStests(test = "all")` (named `lm.LMtests` before spdep 1.3).
///
/// Fits OLS of `y` on `x` (pivoted QR; aliased columns are dropped as `lm` drops them), then
/// with u the residuals, ŷ the fitted values, s² = u'u / n and T = tr(W'W + W²):
///
/// ```text
/// RSerr    = (u'Wu / s²)² / T
/// RSlag    = (u'Wy / s²)² / nJ,   nJ = (||M Wŷ||² + T s²) / s²   (M the residual maker of x)
/// adjRSerr = (u'Wu/s² - T/nJ u'Wy/s²)² / (T (1 - T/nJ))
/// adjRSlag = (u'Wy/s² - u'Wu/s²)² / (nJ - T)
/// SARMA    = adjRSlag + RSerr                                     (df 2)
/// ```
///
/// (Anselin 1988; Anselin, Bera, Florax & Yoon 1996). The fields carry spdep's tests:
/// `lm_error` = RSerr, `lm_lag` = RSlag, `rlm_error` = adjRSerr, `rlm_lag` = adjRSlag,
/// `lm_sarma` = SARMA. As spdep, the tests assume a row-standardised W and warn otherwise.
///
/// # Arguments
///
/// * `y` - Dependent variable
/// * `x` - Design matrix (including the intercept column if the model has one)
/// * `listw` - Spatial weights
///
/// # Errors
///
/// - `InvalidSpecification`: lengths differ, or `y` or `x` has a non-finite value.
/// - `InsufficientData`: no rows.
/// - `Computation`: the residuals are all zero, the weights have no links (T = 0), or a
///   denominator of a test is not positive; the message names the test.
///
/// # Reference
///
/// Anselin, L. (1988). Spatial Econometrics: Methods and Models.
/// Anselin, L., Bera, A.K., Florax, R., Yoon, M.J. (1996). Simple diagnostic tests
/// for spatial dependence. Regional Science and Urban Economics.
pub fn spatial_lm_tests(
    y: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
) -> EconResult<SpatialLmTests> {
    use crate::linalg::solve::least_squares;

    let n = y.len();
    if n != listw.n() || n != x.nrows() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "spatial_lm_tests: y ({n}), the rows of x ({}) and the weights ({}) must have \
                 the same length",
                x.nrows(),
                listw.n()
            ),
        });
    }
    let fit = least_squares(x.view(), y.view())?;
    let u = &fit.residuals;
    let yhat = &fit.fitted;
    let n_f = n as f64;

    let sigma2 = u.dot(u) / n_f;
    if !(sigma2 > 0.0 && sigma2.is_finite()) {
        return Err(EconError::Computation(
            "spatial_lm_tests: the OLS residuals are all zero (or overflow), so the score tests \
             are undefined"
                .to_string(),
        ));
    }
    let tr_w = listw.trace_w2() + listw.trace_wtw();
    if !(tr_w > 0.0 && tr_w.is_finite()) {
        return Err(EconError::Computation(format!(
            "spatial_lm_tests: tr(W'W + W^2) = {tr_w:e} is not positive: the weights have no links"
        )));
    }

    let du_wu = u.dot(&listw.lag(u)) / sigma2;
    let du_wy = u.dot(&listw.lag(y)) / sigma2;
    // ||M W yhat||^2 = (WXb)'M(WXb), from the residuals of W yhat on x
    let wyhat = listw.lag(yhat);
    let m_wyhat = least_squares(x.view(), wyhat.view())?.residuals;
    let nj = (m_wyhat.dot(&m_wyhat) + tr_w * sigma2) / sigma2;

    let positive = |v: f64, what: &str| -> EconResult<f64> {
        if v > 0.0 && v.is_finite() {
            Ok(v)
        } else {
            Err(EconError::Computation(format!(
                "spatial_lm_tests: the denominator of {what} is not positive ({v:e})"
            )))
        }
    };
    let rs_err = du_wu * du_wu / tr_w;
    let rs_lag = du_wy * du_wy / positive(nj, "RSlag")?;
    let adj_err_den = positive(tr_w * (1.0 - tr_w / nj), "adjRSerr")?;
    let adj_rs_err = (du_wu - tr_w / nj * du_wy).powi(2) / adj_err_den;
    let adj_rs_lag = (du_wy - du_wu).powi(2) / positive(nj - tr_w, "adjRSlag")?;
    let sarma = adj_rs_lag + rs_err;

    let test = |statistic: f64, df: usize| LmTestResult {
        statistic,
        df,
        p_value: chi_squared_p_value(statistic, df as f64),
    };

    let mut warnings = Vec::new();
    let off = (0..n)
        .map(|i| listw.get_weights(i))
        .filter(|sw| !sw.weights.is_empty())
        .any(|sw| (sw.weights.iter().sum::<f64>() - 1.0).abs() > 1e-10);
    if off {
        warnings.push(
            "W is not row-standardised (spdep: \"Spatial weights matrix not row standardized\"); \
             the score tests assume row sums of 1"
                .to_string(),
        );
    }

    Ok(SpatialLmTests {
        lm_lag: test(rs_lag, 1),
        lm_error: test(rs_err, 1),
        rlm_lag: test(adj_rs_lag, 1),
        rlm_error: test(adj_rs_err, 1),
        lm_sarma: test(sarma, 2),
        warnings,
    })
}

/// Compute S1 and S2 statistics for spatial weights.
///
/// S1 = 0.5 * sum_i sum_j (w_ij + w_ji)^2
/// S2 = sum_i (sum_j w_ij + sum_j w_ji)^2
///
/// Computed from the sparse rows in O(links): S1 = tr(W'W) + tr(W^2), and S2 from the row
/// sums and the column sums accumulated in one pass.
fn compute_s1_s2(listw: &SpatialWeights) -> (f64, f64) {
    let n = listw.n();

    // S1 = 0.5 * sum (w_ij + w_ji)^2 = tr(W'W) + tr(W^2)
    let s1 = listw.trace_wtw() + listw.trace_w2();

    // S2 = sum_i (row_sum_i + col_sum_i)^2
    let mut row_sums = vec![0.0; n];
    let mut col_sums = vec![0.0; n];
    for (i, row_sum) in row_sums.iter_mut().enumerate() {
        let sw = listw.get_weights(i);
        for (&j, &w) in sw.indices.iter().zip(sw.weights.iter()) {
            *row_sum += w;
            col_sums[j] += w;
        }
    }
    let s2 = row_sums
        .iter()
        .zip(col_sums.iter())
        .map(|(r, c)| (r + c) * (r + c))
        .sum();

    (s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Neighbors;
    use ndarray::array;

    fn col(ds: &crate::data::Dataset, name: &str) -> Vec<f64> {
        crate::data::columns::numeric_column(ds.df(), name, crate::data::columns::NaPolicy::Fail)
            .unwrap()
            .0
    }

    fn coords_of(ds: &crate::data::Dataset) -> Vec<(f64, f64)> {
        col(ds, "cx").into_iter().zip(col(ds, "cy")).collect()
    }

    fn expect_spec(r: EconResult<impl std::fmt::Debug>, needles: &[&str]) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                for needle in needles {
                    assert!(message.contains(needle), "{needle:?} not in {message:?}");
                }
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// R: spdep::moran.test(randomisation = TRUE, zero.policy = TRUE, adjust.n = TRUE), the geary()
    /// statistic and the Cliff-Ord randomisation variance of Geary's C, written out
    /// (spatial_ml_weights.R, case spatial_weights_moran; the variance is checked there against an
    /// exact permutation variance). p2a reports the randomisation variance only, so var_norm is
    /// not compared.
    #[test]
    fn test_validate_moran_islands() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("spatial_weights_moran");
        let coords = coords_of(&ds);
        let x1 = Array1::from(col(&ds, "x1"));
        let n = coords.len();
        let style = crate::spatial::WeightStyle::RowStd;
        let knn =
            SpatialWeights::from_neighbors(&Neighbors::try_from_knn(&coords, 4).unwrap(), style);
        let band = SpatialWeights::from_neighbors(
            &Neighbors::try_from_distance(&coords, 0.0, 0.12, None).unwrap(),
            style,
        );
        let stats = ["I", "E", "var_rand", "z_rand"];
        let ps = ["p_rand", "p_two_sided"];
        let mut chk = Check::new(&case);
        for (key, w, g_c, g_var, g_z, g_p) in [
            (
                "moran",
                &knn,
                "geary_C",
                "geary_var_rand",
                "geary_z",
                "geary_p",
            ),
            (
                "moran_islands",
                &band,
                "geary_C_islands",
                "geary_var_rand_islands",
                "geary_z_islands",
                "geary_p_islands",
            ),
        ] {
            let g = moran_test(&x1, w, MoranAlternative::Greater).unwrap();
            let t = moran_test(&x1, w, MoranAlternative::TwoSided).unwrap();
            let c = geary_test(&x1, w).unwrap();
            assert_eq!(g.n, n);
            chk.named_subset(
                key,
                &stats,
                &stats,
                &[g.statistic, g.expectation, g.variance, g.z_score],
                Tol::stat(n),
            )
            .named_subset(key, &ps, &ps, &[g.p_value, t.p_value], Tol::p_value(n))
            .scalar(g_c, c.statistic, Tol::stat(n))
            .scalar(g_var, c.variance, Tol::stat(n))
            .scalar(g_z, c.z_score, Tol::stat(n))
            .scalar(g_p, c.p_value, Tol::p_value(n));
        }
        let mi = moran_test(&x1, &band, MoranAlternative::Greater).unwrap();
        chk.scalar("n_islands", mi.n_islands as f64, Tol::exact());
        let knn_islands = moran_test(&x1, &knn, MoranAlternative::Greater).unwrap();
        assert_eq!(knn_islands.n_islands, 0);
        assert_eq!(geary_test(&x1, &band).unwrap().n_islands, 4);
        chk.finish();
    }

    #[test]
    fn test_moran_and_geary_refuse_degenerate_inputs() {
        let w_of = |lists: Vec<Vec<usize>>| {
            SpatialWeights::from_neighbors(
                &Neighbors::from_indices(lists),
                crate::spatial::WeightStyle::RowStd,
            )
        };
        let ring = |n: usize| w_of((0..n).map(|i| vec![(i + 1) % n, (i + n - 1) % n]).collect());
        let both = |x: &Array1<f64>, w: &SpatialWeights| {
            [
                moran_test(x, w, MoranAlternative::Greater).map(|r| r.statistic),
                geary_test(x, w).map(|r| r.statistic),
            ]
        };
        let short = |r: &EconResult<f64>| {
            matches!(
                r,
                Err(EconError::InsufficientData {
                    required: 4,
                    provided: 3,
                    ..
                })
            )
        };
        // n = 3: the randomisation variance divides by (n - 1)(n - 2)(n - 3)
        for r in both(&array![1.0, 2.5, 1.7], &ring(3)) {
            assert!(short(&r), "{r:?}");
        }
        // 3 linked units and 3 islands: n_adj = 3 < 4
        let part = w_of(vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![],
            vec![],
            vec![],
        ]);
        let x6 = array![1.0, 2.5, 1.7, 0.4, 3.1, 2.2];
        for r in both(&x6, &part) {
            assert!(short(&r), "{r:?}");
        }
        // every unit an island: S0 = 0
        for r in both(&x6, &w_of(vec![vec![]; 6])) {
            expect_spec(r, &["no links", "S0 = 0"]);
        }
        // constant and non-finite data
        for r in both(&Array1::from_elem(6, 2.0), &ring(6)) {
            expect_spec(r, &["constant"]);
        }
        for r in both(&array![1.0, f64::NAN, 1.7, 0.4, 3.1, 2.2], &ring(6)) {
            expect_spec(r, &["non-finite"]);
        }
        // degenerate permutation distribution: x = (0, 0, 0, 1) on a 4-ring has variance exactly 0
        // (spdep: NaN and a warning)
        match moran_test(
            &array![0.0, 0.0, 0.0, 1.0],
            &ring(4),
            MoranAlternative::Greater,
        ) {
            Err(EconError::Computation(m)) => assert!(m.contains("variance"), "{m}"),
            other => panic!("expected Computation, got {other:?}"),
        }
    }

    #[test]
    fn geary_refuses_a_degenerate_permutation_distribution() {
        let ring4 = SpatialWeights::from_neighbors(
            &Neighbors::from_indices((0..4).map(|i| vec![(i + 1) % 4, (i + 3) % 4]).collect()),
            crate::spatial::WeightStyle::RowStd,
        );
        match geary_test(&array![0.0, 0.0, 0.0, 1.0], &ring4) {
            Err(EconError::Computation(m)) => assert!(m.contains("variance"), "{m}"),
            other => panic!("expected Computation, got {other:?}"),
        }
    }

    /// Var(C) scales as S1/S0^2 (about 2/(n k) for a k-regular binary W), so the guard is
    /// relative to it: a large sparse W has a small but valid variance.
    #[test]
    fn geary_variance_guard_is_relative_to_s1_over_s0_squared() {
        // n k = 1e9: S0 = 1e9, S1 = 2e9, S1/S0^2 = 2e-9
        let (s0, s1) = (1e9, 2e9);
        assert!(!geary_variance_degenerate(1e-9, s0, s1));
        assert!(geary_variance_degenerate(1e-18, s0, s1));
        assert!(geary_variance_degenerate(0.0, s0, s1));
        assert!(geary_variance_degenerate(-1e-9, s0, s1));
        assert!(geary_variance_degenerate(f64::NAN, s0, s1));
    }

    #[test]
    fn test_moran_alternative_parses_aliases_and_rejects_unknown() {
        MoranAlternative::choices().validate().unwrap();
        for (s, v) in [
            ("two.sided", MoranAlternative::TwoSided),
            ("two-sided", MoranAlternative::TwoSided),
            ("Two_Sided", MoranAlternative::TwoSided),
            ("twosided", MoranAlternative::TwoSided),
            ("greater", MoranAlternative::Greater),
            ("positive", MoranAlternative::Greater),
            ("LESS", MoranAlternative::Less),
            ("negative", MoranAlternative::Less),
        ] {
            assert_eq!(s.parse::<MoranAlternative>().unwrap(), v, "{s}");
        }
        expect_spec(
            "up".parse::<MoranAlternative>(),
            &["alternative", "two.sided, greater, less"],
        );
        assert_eq!(MoranAlternative::TwoSided.to_string(), "two.sided");
    }

    fn test_weights() -> SpatialWeights {
        // Simple 4-point structure
        let nb = Neighbors::from_indices(vec![vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2]]);
        SpatialWeights::from_neighbors(&nb, crate::spatial::WeightStyle::RowStd)
    }

    #[test]
    fn test_moran_test_residuals_refuses_df_below_one() {
        let ring3 = SpatialWeights::from_neighbors(
            &Neighbors::from_indices(vec![vec![1, 2], vec![0, 2], vec![0, 1]]),
            crate::spatial::WeightStyle::RowStd,
        );
        let e = Array1::from_vec(vec![0.1, -0.2, 0.1]);
        // rank 3: N - p = 0 (the rank counts, not the columns: lm drops aliased ones)
        let x = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.2, 0.5, 1.1, 1.0, 1.3, -0.4, 0.7, 1.0, -0.8, 0.9, 2.0],
        )
        .unwrap();
        match moran_test_residuals(&e, &x, &ring3, MoranAlternative::Greater) {
            Err(EconError::InsufficientData {
                required: 4,
                provided: 3,
                ..
            }) => {}
            other => panic!("expected InsufficientData {{ 4, 3 }}, got {other:?}"),
        }
    }

    #[test]
    fn test_moran_positive_autocorr() {
        let listw = test_weights();

        // Values with positive spatial autocorrelation
        // (similar values near each other)
        let x = Array1::from_vec(vec![1.0, 1.2, 1.1, 1.3]);

        let result = moran_test(&x, &listw, MoranAlternative::TwoSided).unwrap();

        // With positive autocorrelation, I should be > E[I]
        assert!(result.statistic > result.expectation);
    }

    #[test]
    fn test_moran_negative_autocorr() {
        let listw = test_weights();

        // Values with negative spatial autocorrelation
        // (dissimilar values near each other, like a checkerboard)
        let x = Array1::from_vec(vec![1.0, 5.0, 5.0, 1.0]);

        let result = moran_test(&x, &listw, MoranAlternative::TwoSided).unwrap();

        // With negative autocorrelation, I should be < E[I]
        assert!(result.statistic < result.expectation);
    }

    #[test]
    fn test_geary() {
        let listw = test_weights();
        let x = Array1::from_vec(vec![1.0, 1.2, 1.1, 1.3]);

        let result = geary_test(&x, &listw).unwrap();

        // For positive autocorrelation, C should be < 1
        assert!(result.statistic < result.expectation);
    }

    #[test]
    fn test_localmoran_basic() {
        let listw = test_weights();

        // Values with clear spatial pattern
        let x = Array1::from_vec(vec![1.0, 1.2, 1.1, 1.3]);

        let result = localmoran(&x, &listw, 0.05, 0).unwrap();

        // Should have 4 local statistics
        assert_eq!(result.local_stats.len(), 4);
        assert_eq!(result.n, 4);

        // Global I should be approximately sum of local I
        // (normalized by sum of weights)
        for obs in &result.local_stats {
            // p-values should be valid
            assert!(obs.p_value >= 0.0 && obs.p_value <= 1.0);
        }
    }

    #[test]
    fn test_localmoran_clusters() {
        // Create a larger grid with clear clusters
        let coords: Vec<(f64, f64)> = (0..5)
            .flat_map(|i| (0..5).map(move |j| (i as f64, j as f64)))
            .collect();

        let nb = Neighbors::from_knn(&coords, 4);
        let listw = SpatialWeights::from_neighbors(&nb, crate::spatial::WeightStyle::RowStd);

        // Create data with a hot spot in the corner and cold spot in opposite corner
        let mut x = vec![0.0; 25];
        // Hot spot (top-left)
        x[0] = 5.0;
        x[1] = 4.5;
        x[5] = 4.8;
        x[6] = 4.2;
        // Cold spot (bottom-right)
        x[24] = -5.0;
        x[23] = -4.5;
        x[19] = -4.8;
        x[18] = -4.2;

        let x_arr = Array1::from_vec(x);
        let result = localmoran(&x_arr, &listw, 0.10, 0).unwrap();

        // Should detect some clusters (with relaxed alpha)
        let _total_clusters =
            result.n_high_high + result.n_low_low + result.n_high_low + result.n_low_high;
        // At minimum, the extreme values should show some pattern
        assert!(result.local_stats[0].i_local > 0.0); // Hot spot area should have positive local I
        assert!(result.local_stats[24].i_local > 0.0); // Cold spot area should have positive local I
    }

    use crate::spatial::WeightStyle;
    use crate::spatial::test_fixtures::{Y, band2, dataset, knn3};

    /// `(a - b).abs() <= tol * max(1, |b|)`.
    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:.17e} vs {b:.17e} (tol {tol:e})"
        );
    }

    /// S1 and S2 from a dense W (the computation before the sparse rewrite).
    fn dense_s1_s2(listw: &SpatialWeights) -> (f64, f64) {
        let n = listw.n();
        let w = listw.to_dense();

        // S1 = 0.5 * sum (w_ij + w_ji)^2
        let mut s1 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let sum = w[[i, j]] + w[[j, i]];
                s1 += sum * sum;
            }
        }
        s1 *= 0.5;

        // S2 = sum_i (row_sum_i + col_sum_i)^2
        let mut s2 = 0.0;
        for i in 0..n {
            let row_sum: f64 = w.row(i).sum();
            let col_sum: f64 = w.column(i).sum();
            let total = row_sum + col_sum;
            s2 += total * total;
        }

        (s1, s2)
    }

    /// tr(WW + W'W) from dense products (the computation before the sparse rewrite).
    fn dense_tr_t(listw: &SpatialWeights) -> f64 {
        let w = listw.to_dense();
        let ww = w.dot(&w);
        let wtw = w.t().dot(&w);
        let t = &ww + &wtw;
        t.diag().sum()
    }

    /// y and the design [1, x1] of the fixture.
    fn fixture_ols() -> (Array1<f64>, Array2<f64>) {
        let mut x = Array2::ones((30, 2));
        for (i, &v) in crate::spatial::test_fixtures::X1.iter().enumerate() {
            x[[i, 1]] = v;
        }
        (Array1::from(Y.to_vec()), x)
    }

    // R 4.3.3 (fixture), W row-standardised knn3, B binary knn3, Wd row-standardised band (0, 2]:
    //   z <- y - mean(y); S0 <- sum(W); I <- n/S0 * sum(W * outer(z, z)) / sum(z^2)
    //   S1 <- 0.5*sum((W + t(W))^2); S2 <- sum((rowSums(W) + colSums(W))^2); K <- n*sum(z^4)/sum(z^2)^2
    //   VI <- (n*(S1*(n^2-3n+3) - n*S2 + 3*S0^2) - K*(S1*(n^2-n) - 2n*S2 + 6*S0^2)) / ((n-1)(n-2)(n-3)S0^2) - EI^2
    //   (spdep::moran.test(randomisation = TRUE) formula; for the band, n in the statistic and
    //   the moments is n - #islands, K over all n)
    //   e <- residuals(lm(y ~ x1)); trT <- sum(diag(W %*% W + t(W) %*% W)); LMerr <- (sum(e * W %*% e) / (sum(e^2)/n))^2 / trT
    #[test]
    fn moran_matches_r_reference() {
        let y = Array1::from(Y.to_vec());
        let r = moran_test(&y, &knn3(WeightStyle::RowStd), MoranAlternative::TwoSided).unwrap();
        assert_close(r.statistic, 0.4845122194473625, 1e-12);
        assert_close(r.expectation, -0.034482758620689655, 1e-15);
        assert_close(r.variance, 0.017150361023634795, 1e-12);
        assert_close(r.z_score, 3.963022355681558, 1e-10);
        let b = moran_test(&y, &knn3(WeightStyle::Binary), MoranAlternative::TwoSided).unwrap();
        assert_close(b.statistic, 0.4845122194473625, 1e-12);
        assert_close(b.variance, 0.017150361023634802, 1e-12);
        // band (0, 2] has isolates 0, 16, 29: spdep adjust.n = TRUE uses n - 3 = 27
        let d = moran_test(&y, &band2(WeightStyle::RowStd), MoranAlternative::TwoSided).unwrap();
        assert_close(d.statistic, 0.40795942824716147, 1e-12);
        assert_close(d.expectation, -0.038461538461538464, 1e-15);
        assert_close(d.variance, 0.01903043780998255, 1e-12);
        assert_close(d.z_score, 3.236087036743305, 1e-10);
        assert_eq!(d.n_islands, 3);
    }

    #[test]
    fn s1_s2_match_r_and_the_dense_computation() {
        for (w, s1, s2) in [
            (
                knn3(WeightStyle::RowStd),
                17.333333333333332,
                126.88888888888887,
            ),
            (knn3(WeightStyle::Binary), 156.0, 1142.0),
            (
                band2(WeightStyle::RowStd),
                15.712222222222222,
                112.31388888888888,
            ),
        ] {
            let (a, b) = compute_s1_s2(&w);
            assert_close(a, s1, 1e-11);
            assert_close(b, s2, 1e-10);
            let (da, db) = dense_s1_s2(&w);
            assert_close(a, da, 1e-11);
            assert_close(b, db, 1e-10);
        }
    }

    /// spatial_sim (n = 80) of a T4 case: y, X = [1, x1, x2] and the coordinates.
    fn sim_design(ds: &crate::data::Dataset) -> (Array1<f64>, Array2<f64>, Vec<(f64, f64)>) {
        let (x1, x2) = (col(ds, "x1"), col(ds, "x2"));
        let mut x = Array2::ones((x1.len(), 3));
        for i in 0..x1.len() {
            x[[i, 1]] = x1[i];
            x[[i, 2]] = x2[i];
        }
        (Array1::from(col(ds, "y")), x, coords_of(ds))
    }

    /// Residuals of lm(y ~ X).
    fn ols_residuals(y: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
        crate::linalg::solve::least_squares(x.view(), y.view())
            .unwrap()
            .residuals
    }

    fn sim_knn4(coords: &[(f64, f64)]) -> SpatialWeights {
        SpatialWeights::from_neighbors(
            &Neighbors::try_from_knn(coords, 4).unwrap(),
            WeightStyle::RowStd,
        )
    }

    fn sim_band(coords: &[(f64, f64)], d: f64, style: WeightStyle) -> SpatialWeights {
        SpatialWeights::from_neighbors(
            &Neighbors::try_from_distance(coords, 0.0, d, None).unwrap(),
            style,
        )
    }

    const MORAN_KEYS: [&str; 4] = ["I", "E", "V", "z"];

    fn check_lm_moran(chk: &mut p2a_validation::Check<'_>, key: &str, r: &MoranResult, n: usize) {
        use p2a_validation::Tol;
        let names = ["I", "E", "V", "z", "p"];
        let vals = [r.statistic, r.expectation, r.variance, r.z_score, r.p_value];
        chk.named_subset(key, &MORAN_KEYS, &names, &vals, Tol::stat(n))
            .named_subset(key, &["p"], &names, &vals, Tol::p_value(n));
    }

    /// R: spdep::lm.morantest(lm(y ~ x1 + x2), nb2listw(knn 4), alternative, zero.policy = TRUE)
    /// (spatial_diag_t4.R, case spatial_lm_diag_t4).
    #[test]
    fn test_validate_lm_morantest_knn_spatial_sim() {
        let (case, ds) = crate::test_support::load_case("spatial_lm_diag_t4");
        let (y, x, coords) = sim_design(&ds);
        let u = ols_residuals(&y, &x);
        let w = sim_knn4(&coords);
        let n = y.len();
        let mut chk = p2a_validation::Check::new(&case);
        chk.vector(
            "residuals",
            u.as_slice().unwrap(),
            p2a_validation::Tol::coef(n),
        );
        for (key, alt) in [
            ("lmmoran_knn_greater", MoranAlternative::Greater),
            ("lmmoran_knn_two_sided", MoranAlternative::TwoSided),
            ("lmmoran_knn_less", MoranAlternative::Less),
        ] {
            let r = moran_test_residuals(&u, &x, &w, alt).unwrap();
            assert_eq!(r.alternative, alt);
            check_lm_moran(&mut chk, key, &r, n);
        }
        chk.finish();
    }

    /// R: lm.morantest on a distance band (0, 0.2] whose row sums vary, style W (asymmetric
    /// after standardisation, so listw2U matters) and B.
    #[test]
    fn test_validate_lm_morantest_band_styles_spatial_sim() {
        let (case, ds) = crate::test_support::load_case("spatial_lm_diag_t4");
        let (y, x, coords) = sim_design(&ds);
        let u = ols_residuals(&y, &x);
        let mut chk = p2a_validation::Check::new(&case);
        for (key, style) in [
            ("lmmoran_band_W", WeightStyle::RowStd),
            ("lmmoran_band_B", WeightStyle::Binary),
        ] {
            let w = sim_band(&coords, 0.2, style);
            let r = moran_test_residuals(&u, &x, &w, MoranAlternative::Greater).unwrap();
            check_lm_moran(&mut chk, key, &r, y.len());
        }
        chk.finish();
    }

    /// R: lm.morantest(zero.policy = TRUE) on the band (0, 0.12] with four islands: N counts
    /// the 76 linked units.
    #[test]
    fn test_validate_lm_morantest_islands_spatial_sim() {
        let (case, ds) = crate::test_support::load_case("spatial_lm_diag_t4");
        let (y, x, coords) = sim_design(&ds);
        let u = ols_residuals(&y, &x);
        let w = sim_band(&coords, 0.12, WeightStyle::RowStd);
        let r = moran_test_residuals(&u, &x, &w, MoranAlternative::Greater).unwrap();
        assert_eq!((r.n, r.n_islands), (80, 4));
        let mut chk = p2a_validation::Check::new(&case);
        check_lm_moran(&mut chk, "lmmoran_islands_W", &r, y.len());
        chk.finish();
    }

    /// A 12 x 12 rook grid, row-standardised.
    fn rook_grid(side: usize) -> SpatialWeights {
        let lists = (0..side * side)
            .map(|i| {
                let (r, c) = (i / side, i % side);
                let mut v = Vec::new();
                if r > 0 {
                    v.push(i - side);
                }
                if c > 0 {
                    v.push(i - 1);
                }
                if c + 1 < side {
                    v.push(i + 1);
                }
                if r + 1 < side {
                    v.push(i + side);
                }
                v
            })
            .collect();
        SpatialWeights::from_neighbors(&Neighbors::from_indices(lists), WeightStyle::RowStd)
    }

    /// Under H0 (independent normal errors) the lm.morantest z-score has mean 0 and variance 1
    /// and the 5% two-sided test rejects about 5% of the time.
    #[test]
    fn lm_morantest_under_h0_has_unit_variance_z() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let w = rook_grid(12);
        let n = w.n();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let mut x = Array2::ones((n, 2));
        for i in 0..n {
            x[[i, 1]] = StandardNormal.sample(&mut rng);
        }
        let reps = 2000;
        let mut zs = Vec::with_capacity(reps);
        for rep in 0..reps {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1000 + rep as u64);
            let y = Array1::from_iter((0..n).map(|i| {
                let e: f64 = StandardNormal.sample(&mut rng);
                1.0 + 0.5 * x[[i, 1]] + e
            }));
            let u = ols_residuals(&y, &x);
            zs.push(
                moran_test_residuals(&u, &x, &w, MoranAlternative::TwoSided)
                    .unwrap()
                    .z_score,
            );
        }
        let mean = zs.iter().sum::<f64>() / reps as f64;
        let var = zs.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (reps - 1) as f64;
        let reject = zs
            .iter()
            .filter(|&&z| normal_p_value_two_sided(z) <= 0.05)
            .count() as f64
            / reps as f64;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((0.85..=1.15).contains(&var), "variance {var}");
        assert!(reject < 0.08, "rejection rate {reject}");
    }

    #[test]
    fn lm_tests_match_r_lm_error_and_dense_traces() {
        let (y, x) = fixture_ols();
        for (w, tr_t, lm_err) in [
            (
                knn3(WeightStyle::RowStd),
                17.333333333333332,
                16.996459496295262,
            ),
            (knn3(WeightStyle::Binary), 156.0, 16.996459496295262),
            (
                band2(WeightStyle::RowStd),
                15.712222222222223,
                13.374603292722762,
            ),
        ] {
            assert_close(w.trace_w2() + w.trace_wtw(), tr_t, 1e-12);
            assert_close(dense_tr_t(&w), tr_t, 1e-12);
            let r = spatial_lm_tests(&y, &x, &w).unwrap();
            assert_close(r.lm_error.statistic, lm_err, 1e-9);
            assert_sarma_identities(&r);
        }
    }

    /// SARMA = RSerr + adjRSlag = RSlag + adjRSerr (Anselin et al. 1996).
    fn assert_sarma_identities(r: &SpatialLmTests) {
        let s = r.lm_sarma.statistic;
        let tol = 1e-10 * s.abs();
        assert!(
            (s - r.lm_error.statistic - r.rlm_lag.statistic).abs() <= tol,
            "SARMA {s} vs RSerr + adjRSlag {}",
            r.lm_error.statistic + r.rlm_lag.statistic
        );
        assert!(
            (s - r.lm_lag.statistic - r.rlm_error.statistic).abs() <= tol,
            "SARMA {s} vs RSlag + adjRSerr {}",
            r.lm_lag.statistic + r.rlm_error.statistic
        );
    }

    /// The five statistics and p-values in spdep's order: RSerr, RSlag, adjRSerr, adjRSlag,
    /// SARMA.
    fn rs_values(r: &SpatialLmTests) -> ([f64; 5], [f64; 5]) {
        let t = [
            &r.lm_error,
            &r.lm_lag,
            &r.rlm_error,
            &r.rlm_lag,
            &r.lm_sarma,
        ];
        (t.map(|t| t.statistic), t.map(|t| t.p_value))
    }

    const RS_NAMES: [&str; 5] = ["RSerr", "RSlag", "adjRSerr", "adjRSlag", "SARMA"];

    /// R: spdep::lm.RStests(lm(y ~ x1 + x2), nb2listw(knn 4), test = "all").
    #[test]
    fn test_validate_lm_rstests_knn_spatial_sim() {
        use p2a_validation::Tol;
        let (case, ds) = crate::test_support::load_case("spatial_lm_diag_t4");
        let (y, x, coords) = sim_design(&ds);
        let r = spatial_lm_tests(&y, &x, &sim_knn4(&coords)).unwrap();
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(
            [
                r.lm_error.df,
                r.lm_lag.df,
                r.rlm_error.df,
                r.rlm_lag.df,
                r.lm_sarma.df
            ],
            [1, 1, 1, 1, 2]
        );
        let (stat, p) = rs_values(&r);
        let n = y.len();
        p2a_validation::Check::new(&case)
            .named("rstests_knn_stat", &RS_NAMES, &stat, Tol::stat(n))
            .named("rstests_knn_p", &RS_NAMES, &p, Tol::p_value(n))
            .finish();
    }

    /// R: lm.RStests on the distance band (0, 0.2], style B (spdep warns that W is not row
    /// standardised).
    #[test]
    fn test_validate_lm_rstests_band_binary_spatial_sim() {
        use p2a_validation::Tol;
        let (case, ds) = crate::test_support::load_case("spatial_lm_diag_t4");
        let (y, x, coords) = sim_design(&ds);
        let r = spatial_lm_tests(&y, &x, &sim_band(&coords, 0.2, WeightStyle::Binary)).unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("not row-standardised")),
            "{:?}",
            r.warnings
        );
        let (stat, p) = rs_values(&r);
        let n = y.len();
        p2a_validation::Check::new(&case)
            .named("rstests_band_B_stat", &RS_NAMES, &stat, Tol::stat(n))
            .named("rstests_band_B_p", &RS_NAMES, &p, Tol::p_value(n))
            .finish();
    }

    /// The score tests do not depend on the units of a regressor.
    #[test]
    fn lm_rstests_are_invariant_to_regressor_scale() {
        let (_, ds) = crate::test_support::load_case("spatial_lm_diag_t4");
        let (y, x, coords) = sim_design(&ds);
        let w = sim_knn4(&coords);
        let (base, _) = rs_values(&spatial_lm_tests(&y, &x, &w).unwrap());
        for scale in [1000.0, 1e-3] {
            let mut xs = x.clone();
            xs.column_mut(1).mapv_inplace(|v| v * scale);
            let (stat, _) = rs_values(&spatial_lm_tests(&y, &xs, &w).unwrap());
            for (a, b) in stat.iter().zip(&base) {
                assert!((a - b).abs() <= 1e-9 * b.abs(), "scale {scale}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn lm_rstests_sarma_identities() {
        let (_, ds) = crate::test_support::load_case("spatial_lm_diag_t4");
        let (y, x, coords) = sim_design(&ds);
        for w in [
            sim_knn4(&coords),
            sim_band(&coords, 0.2, WeightStyle::RowStd),
            sim_band(&coords, 0.2, WeightStyle::Binary),
        ] {
            assert_sarma_identities(&spatial_lm_tests(&y, &x, &w).unwrap());
        }
    }

    fn local_columns(r: &LocalMoranResult) -> [Vec<f64>; 5] {
        let col = |f: fn(&LocalMoranObs) -> f64| r.local_stats.iter().map(f).collect::<Vec<_>>();
        [
            col(|o| o.i_local),
            col(|o| o.expectation),
            col(|o| o.variance),
            col(|o| o.z_score),
            col(|o| o.p_value),
        ]
    }

    fn check_localmoran(id: &str, conditional: bool, tag: &str) {
        use p2a_validation::Tol;
        let (case, ds) = crate::test_support::load_case(id);
        let (y, _, coords) = sim_design(&ds);
        let cfg = LocalMoranConfig {
            conditional,
            ..LocalMoranConfig::default()
        };
        let r = localmoran_with(&y, &sim_knn4(&coords), &cfg).unwrap();
        assert_eq!(r.conditional, conditional);
        let n = y.len();
        let [ii, e, v, z, p] = local_columns(&r);
        let mut chk = p2a_validation::Check::new(&case);
        for (key, vals) in [("Ii", &ii), ("E", &e), ("Var", &v), ("Z", &z)] {
            chk.vector(&format!("{tag}_{key}"), vals, Tol::stat(n));
        }
        chk.vector(&format!("{tag}_p"), &p, Tol::p_value(n))
            .scalar("global_I", r.global_i, Tol::stat(n))
            .finish();
    }

    /// R: spdep::localmoran(y, nb2listw(knn 4), conditional = TRUE) (spdep's default; mlvar,
    /// two-sided) (spatial_diag_t4.R, case spatial_localmoran_t4).
    #[test]
    fn test_validate_localmoran_conditional_spatial_sim() {
        check_localmoran("spatial_localmoran_t4", true, "cond");
    }

    /// R: spdep::localmoran(..., conditional = FALSE): total randomisation moments.
    #[test]
    fn test_validate_localmoran_total_spatial_sim() {
        check_localmoran("spatial_localmoran_t4", false, "total");
    }

    #[test]
    fn localmoran_global_i_is_morans_i() {
        let (_, ds) = crate::test_support::load_case("spatial_localmoran_t4");
        let (y, _, coords) = sim_design(&ds);
        for w in [
            sim_knn4(&coords),
            sim_band(&coords, 0.2, WeightStyle::RowStd),
        ] {
            let g = localmoran(&y, &w, 0.05, 0).unwrap().global_i;
            let m = moran_test(&y, &w, MoranAlternative::Greater)
                .unwrap()
                .statistic;
            assert!((g - m).abs() <= 1e-12 * m.abs(), "{g} vs {m}");
        }
    }

    /// The conditional permutation (neighbours drawn without repeats from the other n - 1
    /// values) converges to the enumerated limits of the 7-unit fixture.
    #[test]
    fn localmoran_conditional_permutation_converges_to_exact() {
        let (case, ds) = crate::test_support::load_case("spatial_localmoran_t4");
        let (_, _, coords) = sim_design(&ds);
        let x7 = Array1::from(case.vector("perm7_x"));
        let w7 = SpatialWeights::from_neighbors(
            &Neighbors::try_from_knn(&coords[..7], 2).unwrap(),
            WeightStyle::RowStd,
        );
        let nsim = 99_999;
        let cfg = LocalMoranConfig {
            permutations: nsim,
            ..LocalMoranConfig::default()
        };
        let r = localmoran_with(&x7, &w7, &cfg).unwrap();
        let ii = case.vector("perm7_Ii");
        let limit = case.vector("perm7_folded_limit");
        let nf = nsim as f64;
        for i in 0..7 {
            let o = &r.local_stats[i];
            assert!((o.i_local - ii[i]).abs() <= 1e-12 * ii[i].abs().max(1.0));
            let se = (limit[i] * (1.0 - limit[i]) / nf).sqrt();
            assert!(
                (o.p_value - limit[i]).abs() <= 4.0 * se + 2.0 / nf,
                "unit {i}: folded p {} vs limit {} (4 SE = {})",
                o.p_value,
                limit[i],
                4.0 * se
            );
        }
    }

    /// The one-sided count r_i / nsim converges to the enumerated P(I_perm >= I_i) of the
    /// 7-unit fixture: the folded check alone would miss a count taken in the wrong direction.
    #[test]
    fn localmoran_permutation_count_converges_to_one_sided_limit() {
        let (case, ds) = crate::test_support::load_case("spatial_localmoran_t4");
        let (_, _, coords) = sim_design(&ds);
        let x7 = Array1::from(case.vector("perm7_x"));
        let w7 = SpatialWeights::from_neighbors(
            &Neighbors::try_from_knn(&coords[..7], 2).unwrap(),
            WeightStyle::RowStd,
        );
        let mean = x7.sum() / 7.0;
        let z = x7.mapv(|v| v - mean);
        let m2 = z.iter().map(|v| v * v).sum::<f64>() / 7.0;
        let wz = w7.lag(&z);
        let observed: Vec<f64> = (0..7).map(|i| z[i] / m2 * wz[i]).collect();
        let nsim = 99_999;
        let count = local_moran_permutation_counts(&z, m2, &observed, &w7, nsim, 42).unwrap();
        let p_ge = case.vector("perm7_p_ge");
        let nf = nsim as f64;
        for i in 0..7 {
            let p = count[i] as f64 / nf;
            let se = (p_ge[i] * (1.0 - p_ge[i]) / nf).sqrt();
            assert!(
                (p - p_ge[i]).abs() <= 4.0 * se + 1.0 / nf,
                "unit {i}: r/nsim {p} vs P(I >= Ii) {} (4 SE = {})",
                p_ge[i],
                4.0 * se
            );
        }
    }

    /// A unit that lists itself is refused, with or without permutations: the moments and
    /// the draws from the other n - 1 units assume a zero diagonal, and a row of n entries
    /// would leave the permutation an empty range to draw from.
    #[test]
    fn localmoran_refuses_a_self_link() {
        let x = Array1::from(vec![0.3, -1.2, 0.8, 2.1, -0.4]);
        let nb = Neighbors::from_indices(vec![
            vec![0, 1, 2, 3, 4],
            vec![0, 2],
            vec![1, 3],
            vec![2, 4],
            vec![3, 0],
        ]);
        let w = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        for permutations in [0, 99] {
            let cfg = LocalMoranConfig {
                permutations,
                ..LocalMoranConfig::default()
            };
            match localmoran_with(&x, &w, &cfg) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains("unit 1 lists itself"), "{message}")
                }
                other => panic!("permutations = {permutations}: {other:?}"),
            }
        }
    }

    /// With islands, global_i (m2 over all n units) is n / (n - islands) times moran_test's
    /// statistic, which uses the n - islands units with neighbours.
    #[test]
    fn localmoran_global_i_with_islands_scales_by_n_over_n_adj() {
        let x = Array1::from(vec![0.3, -1.2, 0.8, 2.1, -0.4, 1.7, -0.9]);
        let nb = Neighbors::from_indices(vec![
            vec![1, 4],
            vec![0, 2],
            vec![1, 3],
            vec![2, 4],
            vec![3, 0],
            vec![],
            vec![],
        ]);
        let w = SpatialWeights::from_neighbors(&nb, WeightStyle::RowStd);
        let g = localmoran(&x, &w, 0.05, 0).unwrap().global_i;
        let m = moran_test(&x, &w, MoranAlternative::Greater).unwrap();
        assert_eq!(m.n_islands, 2);
        let expected = m.statistic * 7.0 / 5.0;
        assert!(
            (g - expected).abs() <= 1e-12 * expected.abs(),
            "{g} vs {expected}"
        );
    }

    #[test]
    fn localmoran_seed_changes_permutation_p_only() {
        let (_, ds) = crate::test_support::load_case("spatial_localmoran_t4");
        let (y, _, coords) = sim_design(&ds);
        let w = sim_knn4(&coords);
        let run = |seed| {
            localmoran_with(
                &y,
                &w,
                &LocalMoranConfig {
                    permutations: 199,
                    seed,
                    ..LocalMoranConfig::default()
                },
            )
            .unwrap()
        };
        let (a, a2, b) = (run(1), run(1), run(2));
        let [ia, ea, va, za, pa] = local_columns(&a);
        let [ib, eb, vb, zb, pb] = local_columns(&b);
        assert_eq!(pa, local_columns(&a2)[4], "same seed, same p");
        assert_eq!((ia, ea, va, za), (ib, eb, vb, zb));
        assert_ne!(pa, pb);
    }

    #[test]
    fn localmoran_zero_variance_is_nan_not_clamped() {
        // unit 3 is an island; x[1] equals the mean (z_1 = 0)
        let w = SpatialWeights::from_neighbors(
            &Neighbors::from_indices(vec![vec![1, 2], vec![0, 2], vec![0, 1, 4], vec![], vec![2]]),
            WeightStyle::RowStd,
        );
        let x = array![1.0, 2.0, 3.5, 0.5, 3.0];
        for conditional in [true, false] {
            for permutations in [0, 99] {
                let cfg = LocalMoranConfig {
                    conditional,
                    permutations,
                    alpha: 1.0,
                    ..LocalMoranConfig::default()
                };
                let r = localmoran_with(&x, &w, &cfg).unwrap();
                let island = &r.local_stats[3];
                assert!(
                    island.z_score.is_nan() && island.p_value.is_nan(),
                    "{island:?}"
                );
                assert_eq!(island.cluster, LisaCluster::NotSignificant);
                if conditional {
                    let z0 = &r.local_stats[1];
                    assert!(z0.z_score.is_nan() && z0.p_value.is_nan(), "{z0:?}");
                    assert_eq!(z0.cluster, LisaCluster::NotSignificant);
                }
            }
        }
    }

    #[test]
    fn localmoran_permutations_honour_cancellation() {
        let t = crate::resource::CancelToken::new();
        t.cancel();
        let r = crate::resource::with_budget(crate::resource::Budget::new(None, t), || {
            localmoran(
                &Array1::from(Y.to_vec()),
                &knn3(WeightStyle::RowStd),
                0.05,
                99,
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    #[test]
    fn test_localmoran_permutation() {
        let listw = test_weights();
        let x = Array1::from_vec(vec![1.0, 5.0, 2.0, 4.0]);

        // Test with permutation-based inference
        let result = localmoran(&x, &listw, 0.05, 99).unwrap();

        // Should have valid p-values from permutation
        for obs in &result.local_stats {
            assert!(obs.p_value >= 0.0 && obs.p_value <= 1.0);
            // Permutation p-values are bounded by (1/(perm+1), 1)
            assert!(obs.p_value >= 1.0 / 100.0);
        }
    }
}
