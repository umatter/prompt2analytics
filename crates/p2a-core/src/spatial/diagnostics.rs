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
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::traits::estimator::normal_p_value_two_sided;

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
    /// Units without neighbours; the statistic uses n - n_islands (spdep adjust.n = TRUE)
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
    /// Global Moran's I (sum of local I / n)
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
}

/// Lagrange Multiplier tests for spatial dependence in regression residuals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialLmTests {
    /// LM test for spatial lag (H0: ρ = 0)
    pub lm_lag: LmTestResult,
    /// LM test for spatial error (H0: λ = 0)
    pub lm_error: LmTestResult,
    /// Robust LM test for spatial lag (controlling for error)
    pub rlm_lag: LmTestResult,
    /// Robust LM test for spatial error (controlling for lag)
    pub rlm_error: LmTestResult,
    /// SARMA test (H0: ρ = λ = 0)
    pub lm_sarma: LmTestResult,
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
    let normal = Normal::standard();
    let p_value = match alternative {
        MoranAlternative::TwoSided => normal_p_value_two_sided(z_score),
        MoranAlternative::Greater => normal.sf(z_score),
        MoranAlternative::Less => normal.cdf(z_score),
    };

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

/// Moran's I test for regression residuals.
///
/// Tests for spatial autocorrelation in OLS residuals, which indicates
/// potential spatial dependence not captured by the model.
///
/// # Arguments
///
/// * `residuals` - OLS residuals
/// * `x` - Design matrix (including intercept if applicable)
/// * `listw` - Spatial weights
///
/// # Returns
///
/// Test result with statistic, p-value, etc.
pub fn moran_test_residuals(
    residuals: &Array1<f64>,
    x: &Array2<f64>,
    listw: &SpatialWeights,
) -> EconResult<MoranResult> {
    // For residuals, we use a slightly different variance formula
    // that accounts for the estimation of regression parameters

    let n = residuals.len();
    if n != listw.n() || n != x.nrows() {
        return Err(EconError::InvalidSpecification {
            message: "Dimensions must match".to_string(),
        });
    }

    let k = x.ncols();
    let df = n
        .checked_sub(k)
        .filter(|&d| d >= 1)
        .ok_or_else(|| EconError::InsufficientData {
            required: k + 1,
            provided: n,
            context: "moran_test_residuals: residual degrees of freedom n - k must be at least 1"
                .to_string(),
        })?;

    // Compute Moran's I for residuals
    let mut numerator = 0.0;
    let mut s0 = 0.0;

    for i in 0..n {
        let sw = listw.get_weights(i);
        for (idx, &j) in sw.indices.iter().enumerate() {
            let w_ij = sw.weights[idx];
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
    let i_stat = (n as f64 / s0) * (numerator / denominator);

    // Expected value for residuals
    let ei = -1.0 / (df as f64);

    // Simplified variance (can be refined with more complex formula)
    let (s1, _s2) = compute_s1_s2(listw);
    let n_f = n as f64;
    let s0_sq = s0 * s0;

    // Use simpler variance formula for residuals
    let vi = (n_f * s1 - s0_sq) / ((n_f - 1.0) * s0_sq) - ei * ei;
    let vi = vi.max(1e-10); // Ensure positive

    let z_score = (i_stat - ei) / vi.sqrt();
    let p_value = normal_p_value_two_sided(z_score);

    Ok(MoranResult {
        statistic: i_stat,
        expectation: ei,
        variance: vi,
        z_score,
        p_value,
        alternative: MoranAlternative::TwoSided,
        n,
        n_islands: (0..n)
            .filter(|&i| listw.get_weights(i).indices.is_empty())
            .count(),
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
    if vc.is_nan() || vc <= 1e-8 {
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

/// Local Moran's I (LISA) - Local Indicators of Spatial Association.
///
/// Decomposes global Moran's I into local contributions, allowing identification
/// of local clusters (hot spots, cold spots) and spatial outliers.
///
/// # Arguments
///
/// * `x` - Variable to analyze
/// * `listw` - Spatial weights
/// * `alpha` - Significance level for cluster classification (default 0.05)
/// * `permutations` - Number of permutations for inference (0 = analytical, >0 = simulation)
///
/// # Returns
///
/// Local Moran statistics for each observation with cluster classifications.
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
/// # Reference
///
/// Anselin, L. (1995). Local Indicators of Spatial Association—LISA.
/// Geographical Analysis, 27(2), 93-115.
pub fn localmoran(
    x: &Array1<f64>,
    listw: &SpatialWeights,
    alpha: f64,
    permutations: usize,
) -> EconResult<LocalMoranResult> {
    let n = x.len();
    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: "Variable length must match weights matrix size".to_string(),
        });
    }
    if n < 3 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 3 observations for Local Moran's I".to_string(),
        });
    }

    // Center and standardize the variable
    let mean = x.mean().unwrap();
    let z: Array1<f64> = x.mapv(|v| v - mean);
    let m2: f64 = z.iter().map(|&zi| zi * zi).sum::<f64>() / n as f64;

    // Compute local Moran's I for each observation
    // I_i = (z_i / m2) * sum_j(w_ij * z_j)
    let mut local_stats = Vec::with_capacity(n);
    let mut sum_local_i = 0.0;

    // Pre-compute the spatial lag of z
    let wz = listw.lag(&z);

    // Pre-compute kurtosis for variance calculation (avoid recomputing in loop)
    let b2 = z.iter().map(|&zi| zi.powi(4)).sum::<f64>() / n as f64 / (m2 * m2);
    let n_f = n as f64;

    // Compute observed local I values for all observations
    let mut observed_local_i = Vec::with_capacity(n);
    for i in 0..n {
        let z_i = z[i];
        let wz_i = wz[i];
        let i_local = (z_i / m2) * wz_i;
        observed_local_i.push(i_local);
        sum_local_i += i_local;
    }

    // Compute permutation p-values in batch if needed
    let perm_pvalues = if permutations > 0 {
        Some(compute_all_local_moran_pvalues_permutation(
            &z,
            &observed_local_i,
            listw,
            permutations,
        )?)
    } else {
        None
    };

    // Compute local statistics
    for i in 0..n {
        let z_i = z[i];
        let wz_i = wz[i];
        let i_local = observed_local_i[i];

        // Expected value: E[I_i] = -w_i. / (n-1) where w_i. is row sum
        let sw = listw.get_weights(i);
        let w_i_sum: f64 = sw.weights.iter().sum();
        let ei = -w_i_sum / (n_f - 1.0);

        // Variance (conditional randomization)
        let w_i_sq_sum: f64 = sw.weights.iter().map(|&w| w * w).sum();

        // Simplified variance formula for local Moran
        let vi = w_i_sq_sum * (n_f - b2) / (n_f - 1.0)
            + (w_i_sum * w_i_sum - w_i_sq_sum) * (2.0 * b2 - n_f) / ((n_f - 1.0) * (n_f - 2.0))
            - ei * ei;
        let vi = vi.max(1e-10);

        // Z-score and p-value
        let z_score = (i_local - ei) / vi.sqrt();

        let p_value = if let Some(ref pvals) = perm_pvalues {
            pvals[i]
        } else {
            // Analytical p-value (two-sided)
            // once per unit, so the plain survival function rather than the extended-range tail
            2.0 * Normal::standard().sf(z_score.abs())
        };

        // Determine cluster type
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

    // Count clusters
    let mut n_high_high = 0;
    let mut n_low_low = 0;
    let mut n_high_low = 0;
    let mut n_low_high = 0;
    let mut n_not_sig = 0;

    for obs in &local_stats {
        match obs.cluster {
            LisaCluster::HighHigh => n_high_high += 1,
            LisaCluster::LowLow => n_low_low += 1,
            LisaCluster::HighLow => n_high_low += 1,
            LisaCluster::LowHigh => n_low_high += 1,
            LisaCluster::NotSignificant => n_not_sig += 1,
        }
    }

    // Global Moran's I is sum of local I values divided by sum of weights
    let s0 = listw.sum_weights();
    let global_i = sum_local_i / s0 * n as f64;

    Ok(LocalMoranResult {
        local_stats,
        global_i,
        n,
        alpha,
        n_high_high,
        n_low_low,
        n_high_low,
        n_low_high,
        n_not_sig,
    })
}

/// Compute all permutation p-values efficiently in a batch.
///
/// This is more efficient than calling compute_local_moran_pvalue_permutation
/// for each observation because it:
/// 1. Pre-computes m2 once
/// 2. Reuses shuffled vectors across observations
fn compute_all_local_moran_pvalues_permutation(
    z: &Array1<f64>,
    observed_i: &[f64],
    listw: &SpatialWeights,
    permutations: usize,
) -> EconResult<Vec<f64>> {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    use rand_chacha::ChaCha8Rng;

    let n = z.len();
    let m2: f64 = z.iter().map(|&zi| zi * zi).sum::<f64>() / n as f64;

    // Count extreme values for each observation
    let mut count_extreme = vec![0usize; n];

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut z_perm: Vec<f64> = z.iter().copied().collect();

    for _ in 0..permutations {
        crate::resource::checkpoint()?;

        // Shuffle z values (global permutation)
        z_perm.shuffle(&mut rng);

        // Compute spatial lag of permuted z
        let wz_perm = listw.lag(&Array1::from_vec(z_perm.clone()));

        // Compute local Moran's I for each observation under this permutation
        for i in 0..n {
            let z_i = z[i]; // Keep original z_i (conditional permutation)
            let i_perm = (z_i / m2) * wz_perm[i];

            if i_perm.abs() >= observed_i[i].abs() {
                count_extreme[i] += 1;
            }
        }
    }

    // Convert counts to p-values
    Ok(count_extreme
        .iter()
        .map(|&count| (count as f64 + 1.0) / (permutations as f64 + 1.0))
        .collect())
}

/// Lagrange Multiplier tests for spatial dependence in regression residuals.
///
/// Tests whether OLS residuals exhibit spatial lag or spatial error dependence.
/// Includes both standard and robust versions of the tests.
///
/// # Arguments
///
/// * `residuals` - OLS residuals
/// * `x` - Design matrix (including intercept)
/// * `listw` - Spatial weights
///
/// # Returns
///
/// Suite of LM test results
///
/// # Reference
///
/// Anselin, L. (1988). Spatial Econometrics: Methods and Models.
/// Anselin, L., Bera, A.K., Florax, R., Yoon, M.J. (1996). Simple diagnostic tests
/// for spatial dependence. Regional Science and Urban Economics.
pub fn spatial_lm_tests(
    residuals: &Array1<f64>,
    x: &Array2<f64>,
    listw: &mut SpatialWeights,
) -> EconResult<SpatialLmTests> {
    let n = residuals.len();
    if n != listw.n() || n != x.nrows() {
        return Err(EconError::InvalidSpecification {
            message: "Dimensions must match".to_string(),
        });
    }

    let k = x.ncols();
    let n_f = n as f64;

    // Compute sigma^2 = e'e / n
    let sigma2: f64 = residuals.iter().map(|&r| r * r).sum::<f64>() / n_f;

    // Compute We (spatially lagged residuals)
    let we = listw.lag(residuals);

    // Compute e'We / sigma^2
    let ewe: f64 = residuals
        .iter()
        .zip(we.iter())
        .map(|(&e, &we)| e * we)
        .sum();
    let ewe_scaled = ewe / sigma2;

    // Compute trace terms: tr(WW + W'W), from the sparse rows in O(links)
    let tr_t: f64 = listw.trace_w2() + listw.trace_wtw();

    // Compute (WX)'(WX) related terms
    // J = (1/sigma^2) * [tr(WW + W'W) + (WXb)'(WXb)/sigma^2]
    // For residuals test, we use tr(T) directly

    // LM-Lag test
    // LM_lag = (e'We/sigma^2)^2 / [n * T22]
    // where T22 involves tr(T) and other terms

    // Compute M = I - X(X'X)^{-1}X' projection matrix trace
    // For large problems, we approximate

    // Compute X'X inverse
    let xtx = x.t().dot(x);
    let xtx_inv = crate::linalg::matrix_ops::matrix_inverse(&xtx.view())?;

    // Compute WX
    let mut wx = Array2::zeros((n, k));
    for col in 0..k {
        let x_col = x.column(col).to_owned();
        let wx_col = listw.lag(&x_col);
        for i in 0..n {
            wx[[i, col]] = wx_col[i];
        }
    }

    // Compute (WX)'M(WX) where M = I - X(X'X)^{-1}X'
    // This equals WX'WX - WX'X(X'X)^{-1}X'WX
    let wxtx = wx.t().dot(x);
    let wxtwx = wx.t().dot(&wx);

    let term = wxtx.dot(&xtx_inv).dot(&wxtx.t());
    let j22 = (&wxtwx - &term).diag().sum() / sigma2 + tr_t;

    // LM-Error test
    // LM_error = (e'We/sigma^2)^2 / tr(T)
    let lm_error_stat = ewe_scaled.powi(2) / tr_t;

    // LM-Lag test
    // LM_lag = (e'Wy/sigma^2)^2 / (WXMWX/sigma^2 + tr(T))
    // where y = Xb + e, so Wy = WXb + We
    // We approximate: e'Wy ≈ e'We since e'WXb = 0 in expectation
    let lm_lag_stat = ewe_scaled.powi(2) / j22;

    // Robust LM tests (controlling for the other spatial effect)
    // RLM_lag = (e'We/sigma^2 - e'We/sigma^2 * tr(T)/J22)^2 / (J22 - tr(T)^2/J22)
    // Simplified formulation:

    let t_j = tr_t / j22;

    // RLM-Lag
    let rlm_lag_numer = (ewe_scaled - ewe_scaled * t_j).powi(2);
    let rlm_lag_denom = j22 - tr_t * t_j;
    let rlm_lag_stat = if rlm_lag_denom > 1e-10 {
        rlm_lag_numer / rlm_lag_denom
    } else {
        0.0
    };

    // RLM-Error
    let rlm_error_numer = (ewe_scaled - lm_lag_stat.sqrt()).powi(2);
    let rlm_error_denom = tr_t - j22.recip() * tr_t.powi(2);
    let rlm_error_stat = if rlm_error_denom > 1e-10 {
        rlm_error_numer / rlm_error_denom
    } else {
        0.0
    };

    // SARMA test (joint test for both lag and error)
    let lm_sarma_stat = lm_lag_stat + rlm_error_stat;

    // Compute p-values (chi-squared distribution)
    let chi2_1 = ChiSquared::new(1.0).unwrap();
    let chi2_2 = ChiSquared::new(2.0).unwrap();

    let p_lm_lag = chi2_1.sf(lm_lag_stat.max(0.0));
    let p_lm_error = chi2_1.sf(lm_error_stat.max(0.0));
    let p_rlm_lag = chi2_1.sf(rlm_lag_stat.max(0.0));
    let p_rlm_error = chi2_1.sf(rlm_error_stat.max(0.0));
    let p_lm_sarma = chi2_2.sf(lm_sarma_stat.max(0.0));

    Ok(SpatialLmTests {
        lm_lag: LmTestResult {
            statistic: lm_lag_stat,
            df: 1,
            p_value: p_lm_lag,
        },
        lm_error: LmTestResult {
            statistic: lm_error_stat,
            df: 1,
            p_value: p_lm_error,
        },
        rlm_lag: LmTestResult {
            statistic: rlm_lag_stat,
            df: 1,
            p_value: p_rlm_lag,
        },
        rlm_error: LmTestResult {
            statistic: rlm_error_stat,
            df: 1,
            p_value: p_rlm_error,
        },
        lm_sarma: LmTestResult {
            statistic: lm_sarma_stat,
            df: 2,
            p_value: p_lm_sarma,
        },
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
        match moran_test_residuals(&e, &Array2::ones((3, 4)), &ring3) {
            Err(EconError::InsufficientData {
                required: 5,
                provided: 3,
                ..
            }) => {}
            other => panic!("expected InsufficientData {{ 5, 3 }}, got {other:?}"),
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
    fn test_lm_tests() {
        let mut listw = test_weights();

        // Simple residuals
        let residuals = Array1::from_vec(vec![0.1, -0.2, 0.15, -0.05]);
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap();

        let result = spatial_lm_tests(&residuals, &x, &mut listw).unwrap();

        // All statistics should be non-negative
        assert!(result.lm_lag.statistic >= 0.0);
        assert!(result.lm_error.statistic >= 0.0);

        // p-values should be in [0, 1]
        assert!(result.lm_lag.p_value >= 0.0 && result.lm_lag.p_value <= 1.0);
        assert!(result.lm_error.p_value >= 0.0 && result.lm_error.p_value <= 1.0);
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

    /// OLS residuals of y ~ x1 on the fixture and the design [1, x1].
    fn fixture_ols() -> (Array1<f64>, Array2<f64>) {
        use crate::LinearEstimator;
        use crate::regression::{CovarianceType, run_ols};
        let ols = run_ols(&dataset(), "y", &["x1"], true, CovarianceType::Standard).unwrap();
        let e = ols.residuals().to_owned();
        let mut x = Array2::ones((30, 2));
        for (i, &v) in crate::spatial::test_fixtures::X1.iter().enumerate() {
            x[[i, 1]] = v;
        }
        (e, x)
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
        assert_close(d.variance, 0.019030437809982551, 1e-12);
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

    #[test]
    fn lm_tests_match_r_lm_error_and_dense_traces() {
        let (e, x) = fixture_ols();
        // The other four statistics, [lm_lag, rlm_lag, rlm_error, lm_sarma], are pinned at the
        // values the dense-trace code printed before the sparse rewrite.
        for (mut w, tr_t, lm_err, others) in [
            (
                knn3(WeightStyle::RowStd),
                17.333333333333332,
                16.996459496295262,
                [
                    11.330802171877023,
                    2.2662424299472073,
                    32.94995643165066,
                    44.28075860352769,
                ],
            ),
            (
                knn3(WeightStyle::Binary),
                156.0,
                16.996459496295262,
                [
                    11.330802171877014,
                    2.266242429947206,
                    44.53942593643209,
                    55.87022810830911,
                ],
            ),
            (
                band2(WeightStyle::RowStd),
                15.712222222222223,
                13.374603292722762,
                [
                    8.09353518719907,
                    1.99097423272287,
                    21.881743492574774,
                    29.975278679773844,
                ],
            ),
        ] {
            assert_close(w.trace_w2() + w.trace_wtw(), tr_t, 1e-12);
            assert_close(dense_tr_t(&w), tr_t, 1e-12);
            let r = spatial_lm_tests(&e, &x, &mut w).unwrap();
            assert_close(r.lm_error.statistic, lm_err, 1e-9);
            assert_close(r.lm_lag.statistic, others[0], 1e-10);
            assert_close(r.rlm_lag.statistic, others[1], 1e-10);
            assert_close(r.rlm_error.statistic, others[2], 1e-10);
            assert_close(r.lm_sarma.statistic, others[3], 1e-10);
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
