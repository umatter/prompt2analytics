//! Data-Generating Processes with known true parameters.
//!
//! Each DGP returns a Dataset plus the true parameter values so that MC
//! simulations can assess whether estimators recover them correctly.

use ndarray::{Array1, Array2};
use p2a_core::Dataset;
use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// Regression DGPs
// ============================================================================

/// True parameters for a linear regression DGP.
#[derive(Clone, Debug)]
pub struct RegressionDgp {
    /// True coefficients [intercept, β1, β2, ...].
    pub true_coefs: Vec<f64>,
    /// Error standard deviation.
    pub sigma: f64,
}

/// Homoskedastic linear regression: y = β0 + β1*x1 + β2*x2 + ε, ε ~ N(0, σ²).
pub fn dgp_regression_homoskedastic(n: usize, seed: u64) -> (Dataset, RegressionDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let true_coefs = vec![1.0, 0.5, -0.3];
    let sigma = 1.0;

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            true_coefs[0] + true_coefs[1] * x1[i] + true_coefs[2] * x2[i]
                + rng.gen_range(-3.0..3.0) * sigma / 3.0_f64.sqrt()
                // Approximate N(0,σ²) with uniform scaled to same variance
        })
        .collect();

    // Better: use Box-Muller for proper normal errors
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
            true_coefs[0] + true_coefs[1] * x1[i] + true_coefs[2] * x2[i] + sigma * z
        })
        .collect();

    let df = df! {
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("regression data");

    (Dataset::new(df), RegressionDgp { true_coefs, sigma })
}

/// Heteroskedastic regression: y = β0 + β1*x1 + β2*x2 + ε, Var(ε|x) = σ² * x1⁴.
/// Strong heteroskedasticity (variance ratio ~600:1) so that standard OLS SEs
/// are clearly miscalibrated while HC SEs should give correct coverage.
pub fn dgp_regression_heteroskedastic(n: usize, seed: u64) -> (Dataset, RegressionDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let true_coefs = vec![1.0, 0.5, -0.3];
    let sigma = 1.0;

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(0.2..5.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
            let het_sigma = sigma * x1[i] * x1[i]; // variance proportional to x1⁴
            true_coefs[0] + true_coefs[1] * x1[i] + true_coefs[2] * x2[i] + het_sigma * z
        })
        .collect();

    let df = df! {
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("het regression data");

    (Dataset::new(df), RegressionDgp { true_coefs, sigma })
}

// ============================================================================
// Panel DGPs
// ============================================================================

/// True parameters for a panel data DGP.
#[derive(Clone, Debug)]
pub struct PanelDgp {
    pub true_coefs: Vec<f64>,
    pub sigma: f64,
}

/// Panel FE DGP: y_it = α_i + β1*x1_it + β2*x2_it + ε_it.
pub fn dgp_panel_fe(n_entities: usize, n_periods: usize, seed: u64) -> (Dataset, PanelDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let true_coefs = vec![0.5, -0.3]; // β1, β2 (no intercept — absorbed by FE)
    let sigma = 1.0;
    let n = n_entities * n_periods;

    let mut entity = Vec::with_capacity(n);
    let mut time = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for e in 0..n_entities {
        let alpha_i = (e as f64 - n_entities as f64 / 2.0) * 0.5;
        for t in 0..n_periods {
            entity.push(e as i64);
            time.push(t as i64);
            let xi1 = rng.gen_range(-3.0..3.0);
            let xi2 = rng.gen_range(-3.0..3.0);
            x1.push(xi1);
            x2.push(xi2);
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let eps = sigma * (-2.0 * u1.ln()).sqrt() * u2.cos();
            y.push(alpha_i + true_coefs[0] * xi1 + true_coefs[1] * xi2 + eps);
        }
    }

    let df = df! {
        "entity" => entity,
        "time" => time,
        "y" => y,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("panel data");

    (Dataset::new(df), PanelDgp { true_coefs, sigma })
}

// ============================================================================
// Binary outcome DGPs
// ============================================================================

/// True parameters for a binary choice DGP.
#[derive(Clone, Debug)]
pub struct BinaryDgp {
    pub true_coefs: Vec<f64>, // [intercept, β1, β2]
}

/// Logit DGP: P(y=1|x) = Λ(β0 + β1*x1 + β2*x2).
pub fn dgp_logit(n: usize, seed: u64) -> (Dataset, BinaryDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let true_coefs = vec![-0.5, 1.0, -0.5];

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let linear = true_coefs[0] + true_coefs[1] * x1[i] + true_coefs[2] * x2[i];
            let prob = 1.0 / (1.0 + (-linear).exp());
            if rng.gen_range(0.0..1.0) < prob { 1.0 } else { 0.0 }
        })
        .collect();

    let df = df! { "y" => y, "x1" => x1, "x2" => x2 }.expect("logit data");
    (Dataset::new(df), BinaryDgp { true_coefs })
}

/// Probit DGP: P(y=1|x) = Φ(β0 + β1*x1 + β2*x2).
pub fn dgp_probit(n: usize, seed: u64) -> (Dataset, BinaryDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let true_coefs = vec![-0.5, 1.0, -0.5];

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let linear = true_coefs[0] + true_coefs[1] * x1[i] + true_coefs[2] * x2[i];
            // Φ(x) approximation via logistic CDF scaled: Φ(x) ≈ Λ(x * 1.7)
            // More accurate: use Box-Muller latent variable
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
            let y_star = linear + z; // latent variable
            if y_star > 0.0 { 1.0 } else { 0.0 }
        })
        .collect();

    let df = df! { "y" => y, "x1" => x1, "x2" => x2 }.expect("probit data");
    (Dataset::new(df), BinaryDgp { true_coefs })
}

// ============================================================================
// IV / Causal DGPs
// ============================================================================

/// True parameters for an IV DGP.
#[derive(Clone, Debug)]
pub struct IvDgp {
    /// True coefficient on endogenous variable.
    pub beta_endog: f64,
    /// True coefficient on exogenous variable.
    pub beta_exog: f64,
    /// True intercept.
    pub intercept: f64,
}

/// IV DGP with one endogenous variable and one instrument.
/// y = β0 + β1*x_endog + β2*x_exog + u
/// x_endog = π0 + π1*z + v
/// corr(u, v) ≠ 0 (endogeneity), corr(z, u) = 0 (exclusion restriction)
pub fn dgp_iv(n: usize, seed: u64) -> (Dataset, IvDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dgp = IvDgp {
        beta_endog: 0.8,
        beta_exog: 0.5,
        intercept: 1.0,
    };
    let pi1 = 0.6; // instrument strength

    let z: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let x_exog: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Generate correlated errors (u, v)
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let z1 = (-2.0 * u1.ln()).sqrt() * u2.cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * u2.sin();
        let rho = 0.5; // correlation between u and v
        u.push(z1);
        v.push(rho * z1 + (1.0 - rho * rho).sqrt() * z2);
    }

    let x_endog: Vec<f64> = (0..n).map(|i| pi1 * z[i] + v[i]).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| dgp.intercept + dgp.beta_endog * x_endog[i] + dgp.beta_exog * x_exog[i] + u[i])
        .collect();

    let df = df! {
        "y" => y,
        "x_exog" => x_exog,
        "x_endog" => x_endog,
        "instrument" => z,
    }
    .expect("iv data");

    (Dataset::new(df), dgp)
}

/// DiD DGP: canonical 2×2 design with known treatment effect.
#[derive(Clone, Debug)]
pub struct DidDgp {
    pub true_att: f64,
}

pub fn dgp_did(n: usize, seed: u64) -> (Dataset, DidDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let true_att = 2.0;
    let half = n / 2;

    let mut treatment = Vec::with_capacity(n);
    let mut post = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let t = if i < half { 0.0 } else { 1.0 };
        let p = if i % 2 == 0 { 0.0 } else { 1.0 };
        treatment.push(t);
        post.push(p);
        let x = rng.gen_range(-1.0..1.0);
        x1.push(x);
        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let eps = (-2.0 * u1.ln()).sqrt() * u2.cos();
        y.push(1.0 + 0.5 * t + 0.3 * p + true_att * t * p + 0.4 * x + eps);
    }

    let df = df! {
        "y" => y,
        "treatment" => treatment,
        "post" => post,
        "x1" => x1,
    }
    .expect("did data");

    (Dataset::new(df), DidDgp { true_att })
}

// ============================================================================
// Treatment effect DGPs
// ============================================================================

/// True parameters for treatment effect DGPs.
#[derive(Clone, Debug)]
pub struct TreatmentDgp {
    pub true_ate: f64,
}

/// Treatment DGP for IPW/TMLE: binary treatment with confounders.
pub fn dgp_treatment(n: usize, seed: u64) -> (Dataset, TreatmentDgp) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let true_ate = 0.5;

    let x1: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
    let treatment: Vec<f64> = (0..n)
        .map(|i| {
            let prob = 1.0 / (1.0 + (-0.3 * x1[i] - 0.2 * x2[i]).exp());
            if rng.gen_range(0.0..1.0) < prob { 1.0 } else { 0.0 }
        })
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let u1: f64 = rng.gen_range(0.0001..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let eps = (-2.0 * u1.ln()).sqrt() * u2.cos();
            1.0 + true_ate * treatment[i] + 0.3 * x1[i] + 0.2 * x2[i] + eps
        })
        .collect();

    let df = df! {
        "y" => y,
        "treatment" => treatment,
        "x1" => x1,
        "x2" => x2,
    }
    .expect("treatment data");

    (Dataset::new(df), TreatmentDgp { true_ate })
}

// ============================================================================
// Hypothesis test DGPs (null and alternative)
// ============================================================================

/// Generate two independent normal samples (H0: μ1 = μ2).
pub fn dgp_two_sample_null(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n1 = n / 2;
    let n2 = n - n1;
    let s1 = normal_sample(&mut rng, 0.0, 1.0, n1);
    let s2 = normal_sample(&mut rng, 0.0, 1.0, n2);
    (s1, s2)
}

/// Generate two normal samples with different means (H1: μ1 ≠ μ2).
pub fn dgp_two_sample_alt(n: usize, effect_size: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n1 = n / 2;
    let n2 = n - n1;
    let s1 = normal_sample(&mut rng, 0.0, 1.0, n1);
    let s2 = normal_sample(&mut rng, effect_size, 1.0, n2);
    (s1, s2)
}

/// Generate k independent normal groups (H0: all means equal).
pub fn dgp_k_sample_null(n: usize, k: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_per = n / k;
    (0..k).map(|_| normal_sample(&mut rng, 0.0, 1.0, n_per)).collect()
}

/// Generate a single normal sample (for one-sample tests, H0: μ = 0).
pub fn dgp_one_sample_null(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    normal_sample(&mut rng, 0.0, 1.0, n)
}

/// Generate a normal sample (for Shapiro-Wilk under H0: normality).
pub fn dgp_normal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    normal_sample(&mut rng, 0.0, 1.0, n)
}

/// Generate a non-normal sample (for Shapiro-Wilk under H1).
pub fn dgp_nonnormal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Exponential distribution (strongly skewed)
    (0..n)
        .map(|_| {
            let u: f64 = rng.gen_range(0.0001..1.0);
            -u.ln() // Exp(1)
        })
        .collect()
}

/// Generate a 2×2 contingency table under H0 (independence).
pub fn dgp_contingency_null(n: usize, seed: u64) -> [[f64; 2]; 2] {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let p_row = 0.5;
    let p_col = 0.5;
    let mut table = [[0.0f64; 2]; 2];
    for _ in 0..n {
        let r = if rng.gen_range(0.0..1.0) < p_row { 0 } else { 1 };
        let c = if rng.gen_range(0.0..1.0) < p_col { 0 } else { 1 };
        table[r][c] += 1.0;
    }
    table
}

// ---- helpers ----

fn normal_sample(rng: &mut ChaCha8Rng, mu: f64, sigma: f64, n: usize) -> Vec<f64> {
    let mut samples = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let z1 = (-2.0 * u1.ln()).sqrt() * u2.cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * u2.sin();
        samples.push(mu + sigma * z1);
        i += 1;
        if i < n {
            samples.push(mu + sigma * z2);
            i += 1;
        }
    }
    samples
}
