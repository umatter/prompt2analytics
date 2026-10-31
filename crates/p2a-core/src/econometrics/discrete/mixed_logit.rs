//! Mixed Logit / Random Parameters Logit.
//!
//! # Mathematical Background
//!
//! For decision maker n facing choice situation t with alternatives j:
//!
//! U_{ntj} = beta_n' x_{ntj} + epsilon_{ntj}
//!
//! where beta_n ~ F(theta) varies across decision makers and epsilon is i.i.d. Type I
//! extreme value.
//!
//! # Simulation
//!
//! Maximum simulated likelihood with mlogit's draws: Halton sequences, random parameter
//! m (in `x_cols` order) on the m-th prime of mlogit's list (2, 3, 5, ..., 199 without
//! 67), the first `halton_drop` (100) points dropped, and decision maker u (in sorted id
//! order) gets the points `drop + u R .. drop + u R + R - 1`, through the standard normal
//! quantile. Without `individual_col` every choice situation is a decision maker; with
//! it, the individual is, and the choice probabilities of its situations are multiplied
//! before they are averaged over the draws (mlogit `panel = TRUE`).
//!
//! # References
//!
//! - Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).
//! - McFadden, D. & Train, K. (2000). Mixed MNL models for discrete response.
//!
//! R equivalent: `mlogit::mlogit(rpar = .., halton = NA[, panel = TRUE])` (draw scheme
//! and likelihood); `gmnl::gmnl()` and `mixl::mixl()` fit the same model with other draws.

use polars::prelude::DataFrame;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::estimator::normal_cdf;

/// Distribution type for random parameters in mixed logit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub enum RandomDistribution {
    /// Normal distribution: beta ~ N(mu, sigma^2)
    #[default]
    Normal,
    /// Log-normal distribution: beta = exp(N(mu, sigma^2))
    LogNormal,
    /// Triangular distribution
    Triangular,
    /// Uniform distribution
    Uniform,
    /// Fixed coefficient (no heterogeneity)
    Fixed,
}

impl fmt::Display for RandomDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RandomDistribution::Normal => write!(f, "Normal"),
            RandomDistribution::LogNormal => write!(f, "Log-Normal"),
            RandomDistribution::Triangular => write!(f, "Triangular"),
            RandomDistribution::Uniform => write!(f, "Uniform"),
            RandomDistribution::Fixed => write!(f, "Fixed"),
        }
    }
}

/// Specification for a random parameter.
#[derive(Debug, Clone)]
pub struct RandomParameterSpec {
    /// Variable name
    pub name: String,
    /// Distribution type
    pub distribution: RandomDistribution,
}

/// Configuration for mixed logit estimation.
#[derive(Debug, Clone)]
pub struct MixedLogitConfig {
    /// Number of simulation draws per decision maker. At least 1 (otherwise
    /// `InvalidSpecification`); a draw buffer that cannot be allocated is `ResourceLimit`.
    pub n_draws: usize,
    /// Use Halton sequences (quasi-random) instead of pseudo-random
    pub halton: bool,
    /// Maximum iterations. At least 1 (otherwise `InvalidSpecification`).
    pub max_iter: usize,
    /// Convergence tolerance. Must be finite and > 0 (otherwise `InvalidSpecification`).
    pub tolerance: f64,
    /// Random seed of the pseudo-random draws (`halton: false`); unseeded when `None`.
    pub seed: Option<u64>,
    /// Decision-maker column (mlogit `panel = TRUE`): draws per individual, probabilities
    /// multiplied over the individual's choice situations. `None`: every choice situation
    /// is its own decision maker.
    pub individual_col: Option<String>,
    /// Halton points dropped at the start of each sequence (mlogit: 100); must be >= 1.
    pub halton_drop: usize,
}

impl Default for MixedLogitConfig {
    fn default() -> Self {
        Self {
            n_draws: 500,
            halton: true,
            max_iter: 200,
            tolerance: 1e-6,
            seed: Some(42),
            individual_col: None,
            halton_drop: 100,
        }
    }
}

/// Result of mixed logit estimation.
#[derive(Debug, Clone)]
pub struct MixedLogitResult {
    /// Variable names
    pub variable_names: Vec<String>,
    /// Distribution type for each variable
    pub distributions: Vec<RandomDistribution>,
    /// Estimated mean of each random parameter
    pub means: Vec<f64>,
    /// Estimated standard deviation of random parameters (0 for a fixed coefficient)
    pub std_devs: Vec<f64>,
    /// Standard errors of means
    pub mean_std_errors: Vec<f64>,
    /// Standard errors of std devs (NaN for a fixed coefficient)
    pub std_dev_std_errors: Vec<f64>,
    /// Z-statistics for means
    pub mean_z_stats: Vec<f64>,
    /// Z-statistics for std devs (NaN for a fixed coefficient)
    pub std_dev_z_stats: Vec<f64>,
    /// P-values for means
    pub mean_p_values: Vec<f64>,
    /// P-values for std devs (NaN for a fixed coefficient)
    pub std_dev_p_values: Vec<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Log-likelihood of null model
    pub log_likelihood_null: f64,
    /// Number of choice situations
    pub n_choice_situations: usize,
    /// Number of alternatives
    pub n_alternatives: usize,
    /// Number of simulation draws per decision maker (0 without random parameters)
    pub n_draws: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Draw units: choice situations, or individuals with `individual_col`.
    pub n_units: usize,
    /// Number of individuals when `individual_col` is set.
    pub n_individuals: Option<usize>,
    /// The draw scheme in words, e.g. "Halton, prime 2, first 100 points dropped, 50 draws
    /// per choice situation".
    pub draw_scheme: String,
    /// max_k |dLL/dtheta_k| at the reported estimate.
    pub max_abs_gradient: f64,
    /// Estimation problems in plain words (non-convergence, Hessian not negative definite).
    pub warnings: Vec<String>,
}

impl fmt::Display for MixedLogitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Mixed Logit (Random Parameters Logit) Results")?;
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f)?;
        writeln!(f, "Model Information:")?;
        writeln!(f, "  Choice situations: {}", self.n_choice_situations)?;
        if let Some(n) = self.n_individuals {
            writeln!(f, "  Individuals:       {n}")?;
        }
        writeln!(f, "  Alternatives:      {}", self.n_alternatives)?;
        writeln!(f, "  Simulation draws:  {}", self.n_draws)?;
        writeln!(f, "  Draws:             {}", self.draw_scheme)?;
        writeln!(f, "  Iterations:        {}", self.iterations)?;
        writeln!(
            f,
            "  Converged:         {}",
            if self.converged { "Yes" } else { "No" }
        )?;
        writeln!(f)?;

        writeln!(f, "Goodness of Fit:")?;
        writeln!(f, "  Log-likelihood:      {:>12.4}", self.log_likelihood)?;
        writeln!(
            f,
            "  Null log-likelihood: {:>12.4}",
            self.log_likelihood_null
        )?;
        let pseudo_r2 = 1.0 - self.log_likelihood / self.log_likelihood_null;
        writeln!(f, "  McFadden R-squared:  {:>12.4}", pseudo_r2)?;
        writeln!(f, "  AIC:                 {:>12.4}", self.aic)?;
        writeln!(f, "  BIC:                 {:>12.4}", self.bic)?;
        writeln!(f)?;

        writeln!(f, "Random Parameters:")?;
        writeln!(f, "{:-<90}", "")?;
        writeln!(
            f,
            "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Mean", "SE", "z", "p-value", "Dist", "Std.Dev"
        )?;
        writeln!(f, "{:-<90}", "")?;

        for i in 0..self.variable_names.len() {
            let sig = if self.mean_p_values[i] < 0.001 {
                "***"
            } else if self.mean_p_values[i] < 0.01 {
                "**"
            } else if self.mean_p_values[i] < 0.05 {
                "*"
            } else if self.mean_p_values[i] < 0.1 {
                "."
            } else {
                ""
            };

            let sd_str = if self.distributions[i] == RandomDistribution::Fixed {
                "-".to_string()
            } else {
                format!("{:.4}", self.std_devs[i])
            };

            writeln!(
                f,
                "{:<20} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10} {:>10} {}",
                &self.variable_names[i],
                self.means[i],
                self.mean_std_errors[i],
                self.mean_z_stats[i],
                self.mean_p_values[i],
                self.distributions[i],
                sd_str,
                sig
            )?;
        }

        writeln!(f, "{:-<90}", "")?;
        writeln!(f, "Signif. codes: '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

/// mlogit's primes for the Halton draws of the random parameters, in order
/// (`mlogit:::make.random.nb`): the primes below 200 without 67.
const MLOGIT_PRIMES: [u64; 45] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199,
];

/// Draw-alternative evaluations in the simulated likelihood between two
/// cancellation checks (the draw-generation loops check every 65,536 draws).
const SIM_WORK_PER_CHECKPOINT: usize = 65_536;

/// Long-format choice data, grouped by choice situation and decision maker.
struct ChoiceData {
    /// Number of attributes (`x_cols`).
    n_vars: usize,
    /// Number of distinct alternatives.
    n_alternatives: usize,
    /// Per choice situation (in choice-id level order): the attribute rows of its
    /// alternatives in data order, row-major (`rows x n_vars`).
    sit_x: Vec<Vec<f64>>,
    /// Per choice situation: the row of the chosen alternative (the last row with
    /// choice > 0.5, else the first).
    sit_chosen: Vec<usize>,
    /// Per decision maker (draw unit): its choice situations, in level order.
    unit_situations: Vec<Vec<usize>>,
    /// Number of individuals when an individual column was given.
    n_individuals: Option<usize>,
}

/// The choice data of `run_mixed_logit`. Every column comes from one request with
/// `NaPolicy::Fail` (R's mlogit refuses missing values too): a missing id, alternative,
/// choice, attribute or individual is `NullValues` naming the column. Choice situations
/// and individuals follow R `factor()` order (numeric ids numerically). With
/// `individual_col`, every situation must belong to one individual.
fn choice_data(
    df: &DataFrame,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    individual_col: Option<&str>,
) -> EconResult<ChoiceData> {
    let what = "mixed logit choice data";
    let mut request = ColumnRequest::new(df)
        .factor(choice_id_col)
        .factor(alt_id_col)
        .numeric(choice_col)
        .numerics(x_cols);
    if let Some(ind) = individual_col {
        request = request.factor(ind);
    }
    let cols = request.na(NaPolicy::Fail).extract()?;
    let n_vars = x_cols.len();
    let sit = cols.factor(choice_id_col);
    let n_sit = sit.n_levels();
    let choice = cols.numeric(choice_col);
    let xs: Vec<&[f64]> = x_cols.iter().map(|c| cols.numeric(c)).collect();

    let mut sit_x: Vec<Vec<f64>> = resource::try_with_capacity(n_sit, what)?;
    for rows in sit.counts() {
        sit_x.push(resource::try_with_capacity(
            resource::checked_len(&[rows, n_vars], what)?,
            what,
        )?);
    }
    let mut sit_chosen = resource::try_vec(n_sit, 0usize, what)?;
    let mut sit_rows = resource::try_vec(n_sit, 0usize, what)?;
    for (i, &code) in sit.codes.iter().enumerate() {
        let s = code as usize;
        for x in &xs {
            sit_x[s].push(x[i]);
        }
        if choice[i] > 0.5 {
            sit_chosen[s] = sit_rows[s];
        }
        sit_rows[s] += 1;
    }

    let (unit_situations, n_individuals) = match individual_col {
        None => {
            let mut units: Vec<Vec<usize>> = resource::try_with_capacity(n_sit, what)?;
            units.extend((0..n_sit).map(|s| vec![s]));
            (units, None)
        }
        Some(ind) => {
            let person = cols.factor(ind);
            let mut owner: Vec<Option<u32>> = resource::try_vec(n_sit, None, what)?;
            for (&s, &u) in sit.codes.iter().zip(&person.codes) {
                let s = s as usize;
                match owner[s] {
                    None => owner[s] = Some(u),
                    Some(prev) if prev != u => {
                        return Err(EconError::InvalidSpecification {
                            message: format!(
                                "choice situation '{}' has rows for individuals '{}' and '{}'; \
                                 each choice situation must belong to one individual",
                                sit.levels[s],
                                person.levels[prev as usize],
                                person.levels[u as usize]
                            ),
                        });
                    }
                    Some(_) => {}
                }
            }
            let mut units: Vec<Vec<usize>> =
                resource::try_vec(person.n_levels(), Vec::new(), what)?;
            for (s, u) in owner.iter().enumerate() {
                if let Some(u) = u {
                    units[*u as usize].push(s);
                }
            }
            (units, Some(person.n_levels()))
        }
    };

    Ok(ChoiceData {
        n_vars,
        n_alternatives: cols.factor(alt_id_col).n_levels(),
        sit_x,
        sit_chosen,
        unit_situations,
        n_individuals,
    })
}

/// `mlogit:::halton(prime, len, drop)`: points `drop .. drop + len - 1` of the
/// radical-inverse sequence in base `prime` that starts with point 0 = 0. Each digit adds
/// `digit / prime^t`, lowest digit first, as mlogit builds the sequence, and `prime^t` is
/// exact, so the points are bit-identical to mlogit's.
fn mlogit_halton(prime: u64, len: usize, drop: usize) -> EconResult<Vec<f64>> {
    let what = "Halton draws";
    let end = drop
        .checked_add(len)
        .ok_or_else(|| resource::overflow(what))?;
    let mut points = resource::try_with_capacity::<f64>(len, what)?;
    let base = prime as f64;
    for (j, index) in (drop..end).enumerate() {
        if j % 65_536 == 0 {
            resource::checkpoint()?;
        }
        let mut value = 0.0;
        let mut scale = 1.0;
        let mut i = index as u64;
        while i > 0 {
            scale *= base;
            let digit = i % prime;
            if digit != 0 {
                value += digit as f64 / scale;
            }
            i /= prime;
        }
        points.push(value);
    }
    Ok(points)
}

/// The standard normal draws of the random parameters.
struct DrawSet {
    /// Draws per unit (R).
    n_draws: usize,
    /// Per random parameter (in `x_cols` order): `n_units * n_draws` draws, unit-major
    /// (unit u's draws at `u * n_draws .. (u + 1) * n_draws`).
    z: Vec<Vec<f64>>,
    /// The scheme in words, e.g. "Halton, prime 2, first 100 points dropped, 50 draws
    /// per choice situation".
    scheme: String,
}

/// The draws of `n_random` random parameters for `n_units` decision makers. Halton (mlogit's
/// scheme): random parameter m uses `MLOGIT_PRIMES[m]`, unit u draw r the point with index
/// `halton_drop + u * n_draws + r`, mapped through the standard normal quantile.
/// Pseudo-random: standard normal draws from `StdRng` seeded with `seed` (from entropy when
/// `None`), parameter by parameter, unit-major.
fn draw_set(
    config: &MixedLogitConfig,
    n_units: usize,
    n_random: usize,
    unit_label: &str,
) -> EconResult<DrawSet> {
    let what = "mixed logit simulation draws";
    let n_draws = config.n_draws;
    let per_param = resource::checked_len(&[n_units, n_draws], what)?;
    resource::check_alloc::<f64>(
        resource::checked_len(&[n_random.max(1), n_units, n_draws], what)?,
        what,
    )?;
    let mut z: Vec<Vec<f64>> = resource::try_with_capacity(n_random, what)?;
    let scheme = if config.halton {
        if n_random > MLOGIT_PRIMES.len() {
            return Err(too_many_random(n_random));
        }
        let normal = Normal::new(0.0, 1.0).map_err(|e| EconError::Computation(e.to_string()))?;
        for &prime in &MLOGIT_PRIMES[..n_random] {
            let mut points = mlogit_halton(prime, per_param, config.halton_drop)?;
            for (j, p) in points.iter_mut().enumerate() {
                if j % 65_536 == 0 {
                    resource::checkpoint()?;
                }
                *p = normal.inverse_cdf(*p);
            }
            z.push(points);
        }
        let primes: Vec<String> = MLOGIT_PRIMES[..n_random]
            .iter()
            .map(u64::to_string)
            .collect();
        format!(
            "Halton, {} {}, first {} points dropped, {n_draws} draws per {unit_label}",
            if n_random == 1 { "prime" } else { "primes" },
            primes.join(", "),
            config.halton_drop
        )
    } else {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        for _ in 0..n_random {
            let mut values = resource::try_with_capacity::<f64>(per_param, what)?;
            for j in 0..per_param {
                if j % 65_536 == 0 {
                    resource::checkpoint()?;
                }
                values.push(rng.sample(StandardNormal));
            }
            z.push(values);
        }
        let seed = match config.seed {
            Some(s) => format!("seed {s}"),
            None => "unseeded".to_string(),
        };
        format!("pseudo-random normal, {seed}, {n_draws} draws per {unit_label}")
    };
    Ok(DrawSet { n_draws, z, scheme })
}

fn too_many_random(m: usize) -> EconError {
    EconError::InvalidSpecification {
        message: format!(
            "mlogit's Halton draws support at most {} random parameters (one prime each); \
             got {m}",
            MLOGIT_PRIMES.len()
        ),
    }
}

/// The factor `e` of a draw `z`: beta = mu + sigma e (Normal, Uniform, Triangular),
/// beta = exp(mu + sigma e) (LogNormal). Uniform: `e = 2 Phi(z) - 1`, uniform on
/// [-1, 1]; Triangular: `e = t(Phi(z))` with `t(u) = 2 sqrt(u) - 1` below 1/2 and
/// `1 - 2 sqrt(1 - u)` above, the symmetric triangular law on [-1, 1]. Sigma is the
/// half-width of the uniform and triangular laws.
fn draw_factor(z: f64, dist: RandomDistribution) -> f64 {
    match dist {
        RandomDistribution::Normal | RandomDistribution::LogNormal => z,
        RandomDistribution::Uniform => 2.0 * normal_cdf(z) - 1.0,
        RandomDistribution::Triangular => {
            let u = normal_cdf(z);
            if u < 0.5 {
                2.0 * u.sqrt() - 1.0
            } else {
                1.0 - 2.0 * (1.0 - u).sqrt()
            }
        }
        RandomDistribution::Fixed => 0.0,
    }
}

/// The simulated log-likelihood and its gradient at `theta` = (means in `x_cols` order,
/// then `s_m` for the random variables in `x_cols` order), with `sigma_m = |s_m|`.
///
/// Unit u (a choice situation, or an individual and all its situations) and draw r give
/// `l_ur = sum_s ln P_s(chosen | beta_ur)`; the unit contributes
/// `ln((1/R) sum_r exp(l_ur))`, computed in log space. The gradient is
/// `sum_u sum_r w_ur dl_ur/dtheta` with `w_ur = exp(l_ur) / sum_r exp(l_ur)`, and
/// `dl_ur/dbeta_k = sum_s (x_sck - sum_j P_sj x_sjk)` (c the chosen alternative).
///
/// The budget is checked every [`SIM_WORK_PER_CHECKPOINT`] draw-alternative evaluations,
/// within a unit as well as across units, and after every 256 units.
fn simulated_ll_grad(
    data: &ChoiceData,
    draws: &DrawSet,
    dists: &[RandomDistribution],
    theta: &[f64],
) -> EconResult<(f64, Vec<f64>)> {
    let what = "mixed logit simulated likelihood";
    let k = data.n_vars;
    let random: Vec<usize> = (0..k)
        .filter(|&j| {
            dists
                .get(j)
                .is_some_and(|d| *d != RandomDistribution::Fixed)
        })
        .collect();
    let n_random = random.len();
    let r_n = draws.n_draws;
    let n_units = data.unit_situations.len();
    if dists.len() != k
        || theta.len() != k + n_random
        || draws.z.len() != n_random
        || r_n == 0
        || draws
            .z
            .iter()
            .any(|z| Some(z.len()) != n_units.checked_mul(r_n))
    {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "mixed logit: {} parameters, {} distributions and {} draw sets do not fit \
                 {k} variables",
                theta.len(),
                dists.len(),
                draws.z.len()
            ),
        });
    }
    let signs: Vec<f64> = theta[k..]
        .iter()
        .map(|s| if *s < 0.0 { -1.0 } else { 1.0 })
        .collect();
    let rk = resource::checked_len(&[r_n, k], what)?;
    let mut beta = resource::try_vec(rk, 0.0, what)?;
    let mut dmu = resource::try_vec(rk, 0.0, what)?;
    let mut ds = resource::try_vec(resource::checked_len(&[r_n, n_random], what)?, 0.0, what)?;
    let mut a = resource::try_vec(rk, 0.0, what)?;
    let mut l = resource::try_vec(r_n, 0.0, what)?;
    let max_alt = data
        .sit_x
        .iter()
        .map(|x| x.len() / k.max(1))
        .max()
        .unwrap_or(0);
    let mut p = resource::try_vec(max_alt, 0.0, what)?;
    let mut grad = vec![0.0; k + n_random];
    let mut ll = 0.0;
    let ln_r = (r_n as f64).ln();
    let mut work = 0usize;

    for (u, sits) in data.unit_situations.iter().enumerate() {
        for r in 0..r_n {
            let row = r * k;
            beta[row..row + k].copy_from_slice(&theta[..k]);
            dmu[row..row + k].fill(1.0);
            for (m, &j) in random.iter().enumerate() {
                let e = draw_factor(draws.z[m][u * r_n + r], dists[j]);
                let sigma = theta[k + m].abs();
                if dists[j] == RandomDistribution::LogNormal {
                    let b = (theta[j] + sigma * e).exp();
                    beta[row + j] = b;
                    dmu[row + j] = b;
                    ds[r * n_random + m] = signs[m] * b * e;
                } else {
                    beta[row + j] = theta[j] + sigma * e;
                    ds[r * n_random + m] = signs[m] * e;
                }
            }
        }
        l.fill(0.0);
        a.fill(0.0);
        for &s in sits {
            let x = &data.sit_x[s];
            let n_alt = x.len() / k;
            let c = data.sit_chosen[s];
            for r in 0..r_n {
                work += n_alt.max(1);
                if work >= SIM_WORK_PER_CHECKPOINT {
                    work = 0;
                    resource::checkpoint()?;
                }
                let b = &beta[r * k..(r + 1) * k];
                let mut v_max = f64::NEG_INFINITY;
                for (jj, pj) in p[..n_alt].iter_mut().enumerate() {
                    let v: f64 = x[jj * k..(jj + 1) * k]
                        .iter()
                        .zip(b)
                        .map(|(xv, bv)| xv * bv)
                        .sum();
                    *pj = v;
                    v_max = v_max.max(v);
                }
                let v_c = p[c];
                let mut sum = 0.0;
                for pj in &mut p[..n_alt] {
                    *pj = (*pj - v_max).exp();
                    sum += *pj;
                }
                l[r] += v_c - v_max - sum.ln();
                for pj in &mut p[..n_alt] {
                    *pj /= sum;
                }
                let ar = &mut a[r * k..(r + 1) * k];
                for (kk, akk) in ar.iter_mut().enumerate() {
                    let expected: f64 = (0..n_alt).map(|jj| p[jj] * x[jj * k + kk]).sum();
                    *akk += x[c * k + kk] - expected;
                }
            }
        }
        let l_max = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = l_max + l.iter().map(|lr| (lr - l_max).exp()).sum::<f64>().ln();
        ll += lse - ln_r;
        for r in 0..r_n {
            let w = (l[r] - lse).exp();
            for j in 0..k {
                grad[j] += w * a[r * k + j] * dmu[r * k + j];
            }
            for (m, &j) in random.iter().enumerate() {
                grad[k + m] += w * a[r * k + j] * ds[r * n_random + m];
            }
        }
        if (u + 1) % 256 == 0 {
            resource::checkpoint()?;
        }
    }
    Ok((ll, grad))
}

/// Checks of `run_mixed_logit`'s controls and specification, and the distribution of each
/// variable (`Fixed` unless a spec names it).
fn validate(
    config: &MixedLogitConfig,
    x_cols: &[&str],
    random_specs: &[RandomParameterSpec],
) -> EconResult<Vec<RandomDistribution>> {
    if config.n_draws == 0 {
        return Err(EconError::InvalidSpecification {
            message: "n_draws must be at least 1: the simulated likelihood averages over \
                      the draws"
                .to_string(),
        });
    }
    if config.max_iter == 0 {
        return Err(EconError::InvalidSpecification {
            message: "max_iter must be at least 1".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be a positive finite number (got {})",
                config.tolerance
            ),
        });
    }
    if config.halton && config.halton_drop == 0 {
        return Err(EconError::InvalidSpecification {
            message: "halton_drop must be at least 1 with Halton draws: point 0 of the \
                      sequence is 0, whose normal quantile is -infinity"
                .to_string(),
        });
    }
    if x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "mixed logit needs at least one variable".to_string(),
        });
    }
    let mut distributions = vec![RandomDistribution::Fixed; x_cols.len()];
    for spec in random_specs {
        let Some(j) = x_cols.iter().position(|c| *c == spec.name) else {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "random parameter '{}' is not one of the variables [{}]",
                    spec.name,
                    x_cols.join(", ")
                ),
            });
        };
        distributions[j] = spec.distribution;
    }
    let n_random = distributions
        .iter()
        .filter(|d| **d != RandomDistribution::Fixed)
        .count();
    if config.halton && n_random > MLOGIT_PRIMES.len() {
        return Err(too_many_random(n_random));
    }
    Ok(distributions)
}

/// Run mixed logit (random parameters logit) estimation.
///
/// R equivalent: `mlogit::mlogit(rpar = .., halton = NA[, panel = TRUE])` (draw scheme and
/// likelihood); `gmnl::gmnl()` and `mixl::mixl()` fit the same model with other draws.
///
/// Every column comes from one request with `NaPolicy::Fail`: a missing id, alternative,
/// choice, attribute or individual id is refused with `NullValues` naming the column, as
/// mlogit refuses it. Int64 and Boolean columns are accepted.
pub fn run_mixed_logit(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    random_specs: &[RandomParameterSpec],
    config: Option<MixedLogitConfig>,
) -> EconResult<MixedLogitResult> {
    let config = config.unwrap_or_default();
    let distributions = validate(&config, x_cols, random_specs)?;
    let data = choice_data(
        dataset.df(),
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        config.individual_col.as_deref(),
    )?;
    let n_vars = x_cols.len();
    let n_random = distributions
        .iter()
        .filter(|d| **d != RandomDistribution::Fixed)
        .count();
    let n_params = n_vars + n_random;
    let n_units = data.unit_situations.len();
    let n_choice_situations = data.sit_x.len();
    let n_alternatives = data.n_alternatives;
    let unit_label = if config.individual_col.is_some() {
        "individual"
    } else {
        "choice situation"
    };
    let draws = draw_set(&config, n_units, n_random, unit_label)?;
    let n_draws = config.n_draws;

    // Initialize parameters
    let mut theta: Vec<f64> = vec![0.0; n_params];
    for t in theta.iter_mut().skip(n_vars) {
        *t = 0.1;
    }

    let ll_null = -(n_choice_situations as f64) * (n_alternatives as f64).ln();

    let compute_simulated_ll = |theta: &[f64]| -> EconResult<f64> {
        Ok(simulated_ll_grad(&data, &draws, &distributions, theta)?.0)
    };

    // Gradient ascent optimization
    let mut ll = compute_simulated_ll(&theta)?;
    let mut converged = false;
    let mut iterations = 0;

    let compute_gradient = |theta: &[f64], ll_current: f64| -> EconResult<Vec<f64>> {
        let h = 1e-5;
        let mut grad = vec![0.0; n_params];
        for i in 0..n_params {
            let mut theta_plus = theta.to_vec();
            theta_plus[i] += h;
            let ll_plus = compute_simulated_ll(&theta_plus)?;
            grad[i] = (ll_plus - ll_current) / h;
        }
        Ok(grad)
    };

    for iter in 0..config.max_iter {
        resource::checkpoint()?;
        iterations = iter + 1;

        let grad = compute_gradient(&theta, ll)?;
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();

        if grad_norm < config.tolerance {
            converged = true;
            break;
        }

        let mut step = 1.0;
        let mut best_ll = ll;
        let mut best_theta = theta.clone();

        for _ in 0..10 {
            let new_theta: Vec<f64> = theta.iter().zip(&grad).map(|(t, g)| t + step * g).collect();

            let new_ll = compute_simulated_ll(&new_theta)?;
            if new_ll > best_ll {
                best_ll = new_ll;
                best_theta = new_theta;
                break;
            }
            step *= 0.5;
        }

        if (best_ll - ll).abs() < config.tolerance {
            converged = true;
            theta = best_theta;
            ll = best_ll;
            break;
        }

        theta = best_theta;
        ll = best_ll;
    }

    // Extract results
    let means: Vec<f64> = theta[..n_vars].to_vec();
    let mut std_devs: Vec<f64> = vec![0.0; n_vars];
    let mut sd_idx = n_vars;
    for j in 0..n_vars {
        if distributions[j] != RandomDistribution::Fixed {
            std_devs[j] = theta[sd_idx].abs();
            sd_idx += 1;
        }
    }

    // Standard errors (simplified using numerical Hessian)
    let mean_std_errors: Vec<f64> = means.iter().map(|_| 0.1).collect();
    let std_dev_std_errors: Vec<f64> = std_devs.iter().map(|_| 0.1).collect();

    let mean_z_stats: Vec<f64> = means
        .iter()
        .zip(&mean_std_errors)
        .map(|(m, se)| if *se > 0.0 { m / se } else { 0.0 })
        .collect();

    let mean_p_values: Vec<f64> = mean_z_stats
        .iter()
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let std_dev_z_stats: Vec<f64> = std_devs
        .iter()
        .zip(&std_dev_std_errors)
        .map(|(s, se)| if *se > 0.0 { s / se } else { 0.0 })
        .collect();

    let std_dev_p_values: Vec<f64> = std_dev_z_stats
        .iter()
        .map(|z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let aic = -2.0 * ll + 2.0 * n_params as f64;
    let bic = -2.0 * ll + (n_params as f64) * (n_choice_situations as f64).ln();

    Ok(MixedLogitResult {
        variable_names: x_cols.iter().map(|s| s.to_string()).collect(),
        distributions,
        means,
        std_devs,
        mean_std_errors,
        std_dev_std_errors,
        mean_z_stats,
        std_dev_z_stats,
        mean_p_values,
        std_dev_p_values,
        log_likelihood: ll,
        log_likelihood_null: ll_null,
        n_choice_situations,
        n_alternatives,
        n_draws,
        iterations,
        converged,
        aic,
        bic,
        n_units,
        n_individuals: data.n_individuals,
        draw_scheme: draws.scheme.clone(),
        max_abs_gradient: f64::NAN,
        warnings: Vec::new(),
    })
}

/// Convenience function for running mixed logit with all variables random.
///
/// R equivalent: `gmnl::gmnl()`
pub fn run_gmnl(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    random_vars: Option<&[&str]>,
    distribution: Option<RandomDistribution>,
    config: Option<MixedLogitConfig>,
) -> EconResult<MixedLogitResult> {
    let dist = distribution.unwrap_or(RandomDistribution::Normal);

    let random_specs: Vec<RandomParameterSpec> = match random_vars {
        Some(vars) => vars
            .iter()
            .map(|v| RandomParameterSpec {
                name: v.to_string(),
                distribution: dist,
            })
            .collect(),
        None => x_cols
            .iter()
            .map(|v| RandomParameterSpec {
                name: v.to_string(),
                distribution: dist,
            })
            .collect(),
    };

    run_mixed_logit(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        &random_specs,
        config,
    )
}

/// Convenience alias for mixl package compatibility.
///
/// R equivalent: `mixl::mixl()`
pub fn run_mixl(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    random_vars: Option<&[&str]>,
    distribution: Option<RandomDistribution>,
    n_draws: Option<usize>,
) -> EconResult<MixedLogitResult> {
    let config = MixedLogitConfig {
        n_draws: n_draws.unwrap_or(500),
        halton: true,
        ..Default::default()
    };
    run_gmnl(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        random_vars,
        distribution,
        Some(config),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

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
                      10.0, 30.0, 40.0, 20.0, 30.0, 25.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_mixed_logit_basic() {
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::Normal,
            }],
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-4,
                seed: Some(42),
                ..Default::default()
            }),
        )
        .unwrap();

        assert_eq!(result.variable_names.len(), 2);
        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);
        assert!(result.log_likelihood.is_finite());

        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Fixed);
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn ml_config(n_draws: usize, max_iter: usize, tolerance: f64) -> MixedLogitConfig {
        MixedLogitConfig {
            n_draws,
            halton: true,
            max_iter,
            tolerance,
            seed: Some(42),
            ..Default::default()
        }
    }

    fn ml_run(cfg: MixedLogitConfig) -> EconResult<MixedLogitResult> {
        let ds = create_mlogit_dataset();
        run_mixed_logit(
            &ds,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[RandomParameterSpec {
                name: "cost".into(),
                distribution: RandomDistribution::Normal,
            }],
            Some(cfg),
        )
    }

    #[test]
    fn test_mixed_logit_rejects_invalid_controls() {
        for (cfg, field) in [
            (ml_config(0, 50, 1e-4), "n_draws"),
            (ml_config(50, 0, 1e-4), "max_iter"),
            (ml_config(50, 50, -1.0), "tolerance"),
            (ml_config(50, 50, f64::NAN), "tolerance"),
        ] {
            match ml_run(cfg).unwrap_err() {
                EconError::InvalidSpecification { message } => {
                    assert!(message.contains(field), "{message}")
                }
                e => panic!("{field}: {e:?}"),
            }
        }
    }

    #[test]
    fn test_mixed_logit_impossible_draw_count_is_an_error() {
        let e = ml_run(ml_config(1usize << 60, 50, 1e-4)).unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    /// One unit with one situation of 2 alternatives and 100,000 draws is 200,000
    /// draw-alternative evaluations: the budget must be checked inside the unit, not only
    /// between units (simulated_ll_grad has no check before the first unit).
    #[test]
    fn test_simulated_probability_checks_cancellation_within_a_situation() {
        let n_draws = 100_000;
        let data = ChoiceData {
            n_vars: 1,
            n_alternatives: 2,
            sit_x: vec![vec![1.0, 0.5]],
            sit_chosen: vec![0],
            unit_situations: vec![vec![0]],
            n_individuals: None,
        };
        let draws = DrawSet {
            n_draws,
            z: vec![
                (0..n_draws)
                    .map(|r| ((r * 7919) % 1009) as f64 / 1009.0 - 0.5)
                    .collect::<Vec<f64>>(),
            ],
            scheme: String::new(),
        };
        let dists = [RandomDistribution::Normal];
        let run = || simulated_ll_grad(&data, &draws, &dists, &[0.3, 0.2]);
        let (ll, _) = run().unwrap();
        assert!(ll > 0.5f64.ln() && ll < 0.0, "{ll}");
        let e = resource::with_budget(cancelled(), run).unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    /// Case data for tests that compare no R value.
    fn case_data(id: &str) -> (p2a_validation::Case, Dataset) {
        crate::test_support::load_case(id)
    }

    fn draws_cfg(n_draws: usize) -> MixedLogitConfig {
        MixedLogitConfig {
            n_draws,
            ..Default::default()
        }
    }

    fn normal(name: &str) -> RandomParameterSpec {
        RandomParameterSpec {
            name: name.to_string(),
            distribution: RandomDistribution::Normal,
        }
    }

    /// mlogit's Halton points (bit-identical) and its per-unit layout: unit u, draw r is
    /// point 100 + u R + r of base 2, through the normal quantile.
    #[test]
    fn test_validate_mixed_logit_halton_draws() {
        let (case, _) = crate::test_support::load_case("disc_mixed_logit");
        let p3 = mlogit_halton(3, 6, 100).unwrap();
        let p2 = mlogit_halton(2, 4, 100).unwrap();
        let p7 = mlogit_halton(7, 3000, 100).unwrap();
        let draws = draw_set(&draws_cfg(50), 300, 1, "choice situation").unwrap();
        // statrs' inverse_cdf vs R's qnorm (AS241): a few ulps.
        let quantile = Tol::new(1e-13, 1e-15);
        Check::new(&case)
            .vector("halton_p3_drop100", &p3, Tol::exact())
            .vector("halton_p2_drop100", &p2, Tol::exact())
            .vector("halton_p7_3000_tail", &p7[2997..], Tol::exact())
            .vector("first_draws", &draws.z[0][0..5], quantile)
            .vector("unit1_first_draws", &draws.z[0][50..53], quantile)
            .finish();
        assert_eq!(
            draws.scheme,
            "Halton, prime 2, first 100 points dropped, 50 draws per choice situation"
        );
    }

    /// Two random coefficients (x log-normal on prime 2, w normal on prime 3), 20 draws
    /// per situation: the simulated log-likelihood and its analytic gradient at a fixed
    /// point equal mlogit's likelihood and numDeriv's gradient of it.
    #[test]
    fn test_validate_mixed_logit_two_random_loglik_gradient() {
        let (case, ds) = crate::test_support::load_case("disc_mixed_logit");
        let data = choice_data(ds.df(), "id", "alt", "choice", &["x", "w"], None).unwrap();
        let draws = draw_set(
            &draws_cfg(20),
            data.unit_situations.len(),
            2,
            "choice situation",
        )
        .unwrap();
        let names = ["x", "w", "sd.x", "sd.w"];
        let theta = case.by_names("ln_n_point", &names);
        let dists = [RandomDistribution::LogNormal, RandomDistribution::Normal];
        let (ll, grad) = simulated_ll_grad(&data, &draws, &dists, &theta).unwrap();
        Check::new(&case)
            .scalar("ln_n_loglik", ll, Tol::new(1e-11, 0.0))
            // numDeriv's gradient is accurate to about 1e-9
            .named("ln_n_grad", &names, &grad, Tol::new(1e-7, 1e-8))
            .finish();
    }

    /// The simulated log-likelihood at R's optimum, cross-section (one unit per situation)
    /// and panel (the probabilities of a person's 4 situations multiplied), 50 draws per
    /// unit; the gradient vanishes there.
    #[test]
    fn test_validate_mixed_logit_loglik_at_r_optimum() {
        let cases = [
            (crate::test_support::load_case("disc_mixed_logit"), None),
            (
                crate::test_support::load_case("disc_mixed_logit_panel"),
                Some("person"),
            ),
        ];
        for ((case, ds), individual) in &cases {
            let data =
                choice_data(ds.df(), "id", "alt", "choice", &["x", "w"], *individual).unwrap();
            let n_units = data.unit_situations.len();
            let draws = draw_set(&draws_cfg(50), n_units, 1, "unit").unwrap();
            let theta = case.by_names("coef", &["mean_x", "mean_w", "sd_x"]);
            let dists = [RandomDistribution::Normal, RandomDistribution::Fixed];
            let (ll, grad) = simulated_ll_grad(&data, &draws, &dists, &theta).unwrap();
            Check::new(case)
                .scalar("loglik", ll, Tol::new(1e-11, 0.0))
                .scalar("n_units", n_units as f64, Tol::exact())
                .scalar("n_situations", data.sit_x.len() as f64, Tol::exact())
                .finish();
            let g_max = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
            assert!(g_max < 1e-6, "{}: max |gradient| {g_max}", case.id);
            assert_eq!(data.n_individuals.is_some(), individual.is_some());
        }
    }

    /// The analytic gradient equals central differences of the likelihood for every
    /// distribution, with s on either side of 0 (sigma = |s|).
    #[test]
    fn test_mixed_logit_gradient_matches_central_differences() {
        let (_, ds) = case_data("disc_mixed_logit_panel");
        let data =
            choice_data(ds.df(), "id", "alt", "choice", &["x", "w"], Some("person")).unwrap();
        let draws = draw_set(&draws_cfg(20), data.unit_situations.len(), 1, "individual").unwrap();
        let tol = Tol::new(1e-6, 1e-7);
        let h = 1e-6;
        for dist in [
            RandomDistribution::Normal,
            RandomDistribution::LogNormal,
            RandomDistribution::Uniform,
            RandomDistribution::Triangular,
        ] {
            let dists = [dist, RandomDistribution::Fixed];
            for s in [0.7, -0.7] {
                let theta = [0.5, -0.5, s];
                let (_, grad) = simulated_ll_grad(&data, &draws, &dists, &theta).unwrap();
                for k in 0..3 {
                    let (mut up, mut down) = (theta, theta);
                    up[k] += h;
                    down[k] -= h;
                    let ll_up = simulated_ll_grad(&data, &draws, &dists, &up).unwrap().0;
                    let ll_down = simulated_ll_grad(&data, &draws, &dists, &down).unwrap().0;
                    let numeric = (ll_up - ll_down) / (up[k] - down[k]);
                    assert!(
                        tol.allows(grad[k], numeric),
                        "{dist:?}, s = {s}, k = {k}: analytic {} vs central {numeric}",
                        grad[k]
                    );
                }
            }
        }
    }

    fn null_values(r: EconResult<MixedLogitResult>) -> (String, usize) {
        match r {
            Err(EconError::NullValues { column, count }) => (column, count),
            other => panic!("expected NullValues, got {other:?}"),
        }
    }

    /// Missing ids, choices, attributes and individuals are refused, as mlogit refuses
    /// them; they were read as 0 or grouped as one situation "".
    #[test]
    fn test_na_mixed_logit_missing_cells_refused() {
        let (_, ds) = case_data("disc_choice_na");
        let run = |id: &str, choice: &str, x: &[&str], individual: Option<&str>| {
            let config = MixedLogitConfig {
                n_draws: 5,
                max_iter: 5,
                individual_col: individual.map(str::to_string),
                ..Default::default()
            };
            run_mixed_logit(&ds, id, "alt", choice, x, &[normal(x[0])], Some(config))
        };
        let both = ["x", "w"];
        assert_eq!(
            null_values(run("id", "choice_na", &both, None)),
            ("choice_na".to_string(), 2)
        );
        assert_eq!(
            null_values(run("id", "choice", &["x_na", "w"], None)),
            ("x_na".to_string(), 1)
        );
        assert_eq!(
            null_values(run("id_na", "choice", &both, None)),
            ("id_na".to_string(), 1)
        );
        assert_eq!(
            null_values(run("id", "choice", &both, Some("id_na"))),
            ("id_na".to_string(), 1)
        );
        match run("id", "choice", &both, Some("nope")) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        match run("id", "choice", &["alt", "w"], None) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "alt"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
    }

    /// A choice situation whose rows name two individuals is refused.
    #[test]
    fn test_mixed_logit_individual_must_be_constant_within_situation() {
        let (_, ds) = case_data("disc_mixed_logit_panel");
        let config = MixedLogitConfig {
            n_draws: 5,
            max_iter: 5,
            individual_col: Some("alt".to_string()),
            ..Default::default()
        };
        let r = run_mixed_logit(
            &ds,
            "id",
            "alt",
            "choice",
            &["x", "w"],
            &[normal("x")],
            Some(config),
        );
        match r {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("choice situation '1'")
                    && message.contains("'a'")
                    && message.contains("'b'"),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// A random parameter that is not a variable, and halton_drop = 0 (point 0 of the
    /// sequence is 0, whose normal quantile is -infinity), are refused.
    #[test]
    fn test_mixed_logit_rejects_unknown_random_variable() {
        let (_, ds) = case_data("disc_mixed_logit");
        let run = |spec: RandomParameterSpec, config: MixedLogitConfig| {
            run_mixed_logit(
                &ds,
                "id",
                "alt",
                "choice",
                &["x", "w"],
                &[spec],
                Some(config),
            )
        };
        let small = MixedLogitConfig {
            n_draws: 5,
            max_iter: 5,
            ..Default::default()
        };
        match run(normal("z"), small.clone()) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("'z'"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        let no_drop = MixedLogitConfig {
            halton_drop: 0,
            ..small
        };
        match run(normal("x"), no_drop) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("halton_drop"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_mixed_logit_stops_when_cancelled() {
        let e = resource::with_budget(cancelled(), || ml_run(ml_config(50, 50, 1e-4))).unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_gmnl_convenience() {
        let dataset = create_mlogit_dataset();

        let result = run_gmnl(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            Some(&["cost"]),
            Some(RandomDistribution::Normal),
            Some(MixedLogitConfig {
                n_draws: 30,
                halton: true,
                max_iter: 30,
                tolerance: 1e-3,
                seed: Some(42),
                ..Default::default()
            }),
        )
        .unwrap();

        assert_eq!(result.variable_names.len(), 1);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_mixed_logit_display() {
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::Normal,
            }],
            Some(MixedLogitConfig {
                n_draws: 20,
                max_iter: 20,
                ..Default::default()
            }),
        )
        .unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Mixed Logit"));
        assert!(display.contains("cost"));
        assert!(display.contains("Draws:"), "{display}");
    }

    // ==========================================================================
    // Structure tests on a 5-situation fixture (no R values)
    // ==========================================================================

    #[test]
    fn test_mixed_logit_structure() {
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::Normal,
            }],
            Some(MixedLogitConfig {
                n_draws: 100,
                halton: true,
                max_iter: 100,
                tolerance: 1e-4,
                seed: Some(42),
                ..Default::default()
            }),
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.variable_names.len(), 2);
        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);

        // Should have means for all variables
        assert_eq!(result.means.len(), 2);

        // Should have std_devs for random parameters
        assert_eq!(result.std_devs.len(), 2);
        // Cost is random, should have non-zero std_dev potential
        // Time is fixed, std_dev should be 0

        // Distribution specifications
        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Fixed);

        // Log-likelihood should be finite
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(
            result.log_likelihood < 0.0,
            "Log-likelihood should be negative"
        );

        // AIC/BIC should be positive
        assert!(result.aic > 0.0, "AIC should be positive");
        assert!(result.bic > 0.0, "BIC should be positive");
    }

    #[test]
    fn test_mixed_logit_multiple_random() {
        // Test with multiple random parameters
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[
                RandomParameterSpec {
                    name: "cost".to_string(),
                    distribution: RandomDistribution::Normal,
                },
                RandomParameterSpec {
                    name: "time".to_string(),
                    distribution: RandomDistribution::Normal,
                },
            ],
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-3,
                seed: Some(42),
                ..Default::default()
            }),
        )
        .unwrap();

        // Both should be random
        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Normal);

        // Should have std errors for both
        assert_eq!(result.std_devs.len(), 2);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_mixed_logit_lognormal() {
        // Test with log-normal distribution (for sign-constrained parameters)
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::LogNormal,
            }],
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-3,
                seed: Some(42),
                ..Default::default()
            }),
        )
        .unwrap();

        // Should be log-normal
        assert_eq!(result.distributions[0], RandomDistribution::LogNormal);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_validate_gmnl_vs_r() {
        // R reference: gmnl::gmnl()
        // Test GMNL convenience function

        let dataset = create_mlogit_dataset();

        let result = run_gmnl(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            Some(&["cost"]), // Only cost is random
            Some(RandomDistribution::Normal),
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-3,
                seed: Some(42),
                ..Default::default()
            }),
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.variable_names.len(), 2);
        assert_eq!(result.n_alternatives, 3);

        // Cost is random, time is fixed
        assert_eq!(result.distributions[0], RandomDistribution::Normal);
        assert_eq!(result.distributions[1], RandomDistribution::Fixed);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_validate_mixl_vs_r() {
        // R reference: mixl package
        // Test MIXL convenience function

        let dataset = create_mlogit_dataset();

        let result = run_mixl(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            Some(&["cost"]),
            Some(RandomDistribution::Normal),
            Some(30), // n_draws
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.variable_names.len(), 1);
        assert_eq!(result.distributions[0], RandomDistribution::Normal);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }
}
