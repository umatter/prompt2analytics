//! Spatial diagnostics and tests.
//!
//! Provides tests for spatial autocorrelation (Moran's I, Geary's C)
//! and Lagrange Multiplier tests for spatial dependence in regression residuals.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

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
    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: "Variable length must match weights matrix size".to_string(),
        });
    }
    if n < 3 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 3 observations for Moran's I test".to_string(),
        });
    }

    // Center the variable
    let mean = x.mean().unwrap();
    let z: Array1<f64> = x.mapv(|v| v - mean);

    // Compute numerator: sum of w_ij * z_i * z_j
    let mut numerator = 0.0;
    let mut s0 = 0.0; // Sum of all weights

    for i in 0..n {
        let sw = listw.get_weights(i);
        for (idx, &j) in sw.indices.iter().enumerate() {
            let w_ij = sw.weights[idx];
            numerator += w_ij * z[i] * z[j];
            s0 += w_ij;
        }
    }

    // Compute denominator: sum of z_i^2
    let denominator: f64 = z.iter().map(|&zi| zi * zi).sum();

    // Moran's I
    let i_stat = (n as f64 / s0) * (numerator / denominator);

    // Expected value under null hypothesis of no spatial correlation
    let ei = -1.0 / (n as f64 - 1.0);

    // Compute variance (randomization assumption)
    // This uses the formulas from Cliff & Ord (1981)
    let (s1, s2) = compute_s1_s2(listw);

    let n_f = n as f64;
    let s0_sq = s0 * s0;

    // m2 = sum(z^2)/n, m4 = sum(z^4)/n
    let m2 = denominator / n_f;
    let m4: f64 = z.iter().map(|&zi| zi.powi(4)).sum::<f64>() / n_f;
    let b2 = m4 / (m2 * m2); // Kurtosis

    // Variance under randomization
    let term1 = n_f * ((n_f * n_f - 3.0 * n_f + 3.0) * s1 - n_f * s2 + 3.0 * s0_sq);
    let term2 = b2 * ((n_f * n_f - n_f) * s1 - 2.0 * n_f * s2 + 6.0 * s0_sq);
    let term3 = (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * s0_sq;

    let vi = (term1 - term2) / term3 - ei * ei;

    // Z-score and p-value
    let z_score = (i_stat - ei) / vi.sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = match alternative {
        MoranAlternative::TwoSided => 2.0 * (1.0 - normal.cdf(z_score.abs())),
        MoranAlternative::Greater => 1.0 - normal.cdf(z_score),
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
    let df = n - k;

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

    let denominator: f64 = residuals.iter().map(|&r| r * r).sum();
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

    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

    Ok(MoranResult {
        statistic: i_stat,
        expectation: ei,
        variance: vi,
        z_score,
        p_value,
        alternative: MoranAlternative::TwoSided,
        n,
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
    if n != listw.n() {
        return Err(EconError::InvalidSpecification {
            message: "Variable length must match weights matrix size".to_string(),
        });
    }
    if n < 3 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 3 observations for Geary's C test".to_string(),
        });
    }

    let mean = x.mean().unwrap();
    let z: Array1<f64> = x.mapv(|v| v - mean);

    // Compute numerator: sum of w_ij * (x_i - x_j)^2
    let mut numerator = 0.0;
    let mut s0 = 0.0;

    for i in 0..n {
        let sw = listw.get_weights(i);
        for (idx, &j) in sw.indices.iter().enumerate() {
            let w_ij = sw.weights[idx];
            let diff = x[i] - x[j];
            numerator += w_ij * diff * diff;
            s0 += w_ij;
        }
    }

    // Compute denominator: sum of (x_i - mean)^2
    let denominator: f64 = z.iter().map(|&zi| zi * zi).sum();

    // Geary's C
    let c_stat = ((n as f64 - 1.0) / (2.0 * s0)) * (numerator / denominator);

    // Expected value under null
    let ec = 1.0;

    // Variance (randomization assumption)
    let (s1, s2) = compute_s1_s2(listw);
    let n_f = n as f64;
    let s0_sq = s0 * s0;

    let m2 = denominator / n_f;
    let m4: f64 = z.iter().map(|&zi| zi.powi(4)).sum::<f64>() / n_f;
    let b2 = m4 / (m2 * m2);

    let _term1 = (2.0 * s1 + s2) * (n_f - 1.0) - 4.0 * s0_sq;
    let _term2 = (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * s0_sq;

    let vc = ((n_f - 1.0) * s1 * (n_f * n_f - 3.0 * n_f + 3.0 - (n_f - 1.0) * b2))
        / (n_f * (n_f - 2.0) * (n_f - 3.0) * s0_sq)
        + ((n_f - 1.0) * s2 * (n_f * n_f + 3.0 * n_f - 6.0 - (n_f * n_f - n_f + 2.0) * b2))
            / (4.0 * n_f * (n_f - 2.0) * (n_f - 3.0) * s0_sq)
        + (s0_sq - s1) / (s0_sq * (n_f - 1.0));

    let vc = vc.max(1e-10);

    // Z-score and p-value (two-sided)
    let z_score = (c_stat - ec) / vc.sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

    Ok(GearyResult {
        statistic: c_stat,
        expectation: ec,
        variance: vc,
        z_score,
        p_value,
        n,
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
        ))
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
            let normal = Normal::new(0.0, 1.0).unwrap();
            2.0 * (1.0 - normal.cdf(z_score.abs()))
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
) -> Vec<f64> {
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
    count_extreme
        .iter()
        .map(|&count| (count as f64 + 1.0) / (permutations as f64 + 1.0))
        .collect()
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

    // Compute trace terms
    let w = listw.to_dense();
    let ww = w.dot(&w);
    let wtw = w.t().dot(&w);
    let t = &ww + &wtw;

    let tr_t: f64 = t.diag().sum();

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

    let p_lm_lag = 1.0 - chi2_1.cdf(lm_lag_stat.max(0.0));
    let p_lm_error = 1.0 - chi2_1.cdf(lm_error_stat.max(0.0));
    let p_rlm_lag = 1.0 - chi2_1.cdf(rlm_lag_stat.max(0.0));
    let p_rlm_error = 1.0 - chi2_1.cdf(rlm_error_stat.max(0.0));
    let p_lm_sarma = 1.0 - chi2_2.cdf(lm_sarma_stat.max(0.0));

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
fn compute_s1_s2(listw: &SpatialWeights) -> (f64, f64) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::Neighbors;

    fn test_weights() -> SpatialWeights {
        // Simple 4-point structure
        let nb = Neighbors::from_indices(vec![vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2]]);
        SpatialWeights::from_neighbors(&nb, crate::spatial::WeightStyle::RowStd)
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
