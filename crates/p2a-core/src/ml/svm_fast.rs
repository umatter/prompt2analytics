//! Support vector classification: LIBSVM's C-SVC solver.
//!
//! The dual `min 1/2 a'Qa - e'a` subject to `0 <= a_i <= C`, `y'a = 0`, with `Q_ij = y_i y_j K_ij`,
//! solved as LIBSVM 3.x solves it (Fan, Chen and Lin 2005; Chang and Lin 2011):
//!
//! - **Second-order working-set selection** (WSS3): the pair that most lowers the objective, among
//!   those violating the optimality condition `m(a) - M(a) <= tolerance`.
//! - **A kernel-column cache**, least recently used first, bounded by `cache_size_mb`. Entries are
//!   single precision, as LIBSVM's `Qfloat`, so a cache holds as many columns as LIBSVM's and the
//!   solution is that of the rounded problem, about 1e-7 relative from the exact one. Missing
//!   entries of a long column are filled in parallel; the values do not depend on the threads.
//! - **Shrinking** with gradient reconstruction, and a `min_working_set` floor on the active set (0 =
//!   LIBSVM's behaviour; at `n` it keeps every variable active).
//!
//! Labels are any two finite values: the smaller is -1, the larger +1, and predictions come back in
//! the original labels.
//!
//! The solver is compared with e1071, not reproduced digit by digit: the rows stay in their original
//! order, so a run takes a different path to the same optimum (`validation/ml/svm.md`).
//!
//! R equivalent: `e1071::svm(type = 'C-classification', scale = FALSE)`
//!
//! # References
//!
//! - Fan, R.-E., Chen, P.-H., & Lin, C.-J. (2005). Working set selection using second order
//!   information for training support vector machines. *JMLR*, 6, 1889-1918.
//! - Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. *ACM TIST*,
//!   2(3), 27.

use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::svm::binary_classes;
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};

/// Fast SVM configuration
#[derive(Debug, Clone)]
pub struct FastSvmConfig {
    /// Regularization parameter; finite and positive
    pub c: f64,
    /// Kernel type
    pub kernel: FastKernel,
    /// LIBSVM's `eps`, the stopping tolerance on `m(a) - M(a)`; finite and positive
    pub tolerance: f64,
    /// Maximum LIBSVM iterations (one pair each); at least 1
    pub max_iter: usize,
    /// Size of the kernel cache in MB, at least 1 (LIBSVM's `cache_size`)
    pub cache_size_mb: usize,
    /// Enable shrinking
    pub shrinking: bool,
    /// Floor on the active set of shrinking (0 = LIBSVM)
    pub min_working_set: usize,
}

impl Default for FastSvmConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: FastKernel::Rbf { gamma: 1.0 },
            tolerance: 1e-3,
            max_iter: 10_000_000,
            cache_size_mb: 100,
            shrinking: true,
            min_working_set: 0,
        }
    }
}

/// Kernel types with precomputed parameters
#[derive(Debug, Clone, Copy)]
pub enum FastKernel {
    Linear,
    Rbf { gamma: f64 },
    Polynomial { gamma: f64, coef0: f64, degree: u32 },
}

/// Fast SVM result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastSvmResult {
    /// Training predictions, in the original labels
    pub predictions: Vec<f64>,
    /// `alpha_i >= 0`, in row order
    pub alphas: Vec<f64>,
    /// `decision(x) = sum_i alpha_i y_i K(x_i, x) + bias`, `y_i = +1` for the larger label
    pub bias: f64,
    /// Rows with `alpha_i > 0`
    pub n_support_vectors: usize,
    /// LIBSVM iterations
    pub n_iterations: usize,
    /// The dual objective `sum alpha - 1/2 alpha'Q alpha`
    pub objective_value: f64,
    /// Decision values on the training rows; positive means the larger label
    #[serde(default)]
    pub decision_values: Vec<f64>,
    /// Support vectors' rows, 0-based, ascending
    #[serde(default)]
    pub support_vector_indices: Vec<usize>,
    /// `alpha_i y_i`, aligned with `support_vector_indices`
    #[serde(default)]
    pub dual_coefficients: Vec<f64>,
    /// `(negative, positive)` = (smaller, larger) label
    #[serde(default)]
    pub class_labels: (f64, f64),
    /// Whether the optimality condition held before `max_iter`
    #[serde(default)]
    pub converged: bool,
    /// The convergence record of the solver
    #[serde(default)]
    pub convergence: Option<Convergence>,
    /// What the caller should know about this fit
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Kernel values computed for the cache
    #[serde(default)]
    pub kernel_evaluations: u64,
    /// Largest number of cached single-precision entries
    #[serde(default)]
    pub cache_peak_entries: usize,
    /// Smallest active set reached by shrinking
    #[serde(default)]
    pub min_active_size: usize,
}

impl ResultNotes for FastSvmResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

impl std::fmt::Display for FastSvmResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let positive = self
            .predictions
            .iter()
            .filter(|p| **p == self.class_labels.1)
            .count();
        writeln!(f, "Support vector classification (LIBSVM C-SVC)")?;
        writeln!(f, "Observations: {}", self.predictions.len())?;
        writeln!(
            f,
            "Classes: {} (-1), {} (+1); predicted +1: {positive}",
            self.class_labels.0, self.class_labels.1
        )?;
        writeln!(f, "Support vectors: {}", self.n_support_vectors)?;
        writeln!(f, "Bias: {:.6}", self.bias)?;
        writeln!(f, "Dual objective: {:.6}", self.objective_value)?;
        writeln!(f, "Iterations: {}", self.n_iterations)?;
        writeln!(
            f,
            "Converged: {}",
            if self.converged { "yes" } else { "no" }
        )?;
        f.write_str(&notes_text(self))
    }
}

/// One column of the LRU list; `prev` and `next` are positions, with the sentinel at `l`.
struct Head {
    data: Vec<f32>,
    len: usize,
    prev: usize,
    next: usize,
}

/// LIBSVM's `Cache`: columns of `Q`, grown on demand, least recently used evicted first. The budget
/// of `max(bytes / 4 - 8 l, 2 l)` entries always holds the two columns of a step.
struct KernelCache {
    heads: Vec<Head>,
    /// Sentinel position in `heads`
    root: usize,
    /// Entries still free
    free: usize,
    budget: usize,
    peak: usize,
}

impl KernelCache {
    fn new(l: usize, size_bytes: usize) -> EconResult<Self> {
        let two_l = resource::checked_len(&[l, 2], "SVM kernel cache")?;
        let eight_l = resource::checked_len(&[l, 8], "SVM kernel cache")?;
        let budget = (size_bytes / 4).saturating_sub(eight_l).max(two_l);
        let mut heads: Vec<Head> = resource::try_with_capacity(l + 1, "SVM kernel cache")?;
        for _ in 0..=l {
            heads.push(Head {
                data: Vec::new(),
                len: 0,
                prev: l,
                next: l,
            });
        }
        Ok(KernelCache {
            heads,
            root: l,
            free: budget,
            budget,
            peak: 0,
        })
    }

    fn lru_delete(&mut self, h: usize) {
        let (prev, next) = (self.heads[h].prev, self.heads[h].next);
        self.heads[prev].next = next;
        self.heads[next].prev = prev;
    }

    fn lru_insert(&mut self, h: usize) {
        let last = self.heads[self.root].prev;
        self.heads[h].next = self.root;
        self.heads[h].prev = last;
        self.heads[last].next = h;
        self.heads[self.root].prev = h;
    }

    fn release(&mut self, h: usize) {
        self.lru_delete(h);
        self.free += self.heads[h].len;
        self.heads[h].data = Vec::new();
        self.heads[h].len = 0;
    }

    /// Column `index` with at least `len` entries, most recently used now. `.0` is the start of the
    /// entries to fill: `len` when nothing is missing.
    fn get_data(&mut self, index: usize, len: usize) -> EconResult<(usize, &mut [f32])> {
        if self.heads[index].len > 0 {
            self.lru_delete(index);
        }
        let old = self.heads[index].len;
        let mut start = len;
        if len > old {
            let more = len - old;
            while self.free < more {
                let oldest = self.heads[self.root].next;
                self.release(oldest);
            }
            let data = &mut self.heads[index].data;
            data.try_reserve_exact(len - data.len())
                .map_err(|_| EconError::ResourceLimit("SVM kernel cache".to_string()))?;
            data.resize(len, 0.0);
            self.heads[index].len = len;
            self.free -= more;
            self.peak = self.peak.max(self.budget - self.free);
            start = old;
        }
        self.lru_insert(index);
        Ok((start, &mut self.heads[index].data[..len]))
    }

    /// A column made sure of by [`KernelCache::get_data`].
    fn column(&self, index: usize) -> &[f32] {
        &self.heads[index].data
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        if self.heads[i].len > 0 {
            self.lru_delete(i);
        }
        if self.heads[j].len > 0 {
            self.lru_delete(j);
        }
        let (a, b) = (i.min(j), i.max(j));
        let (left, right) = self.heads.split_at_mut(b);
        std::mem::swap(&mut left[a].data, &mut right[0].data);
        std::mem::swap(&mut left[a].len, &mut right[0].len);
        if self.heads[i].len > 0 {
            self.lru_insert(i);
        }
        if self.heads[j].len > 0 {
            self.lru_insert(j);
        }
        let mut h = self.heads[self.root].next;
        while h != self.root {
            let next = self.heads[h].next;
            if self.heads[h].len > a {
                if self.heads[h].len > b {
                    self.heads[h].data.swap(a, b);
                } else {
                    // the column stops between the two: give it up
                    self.release(h);
                }
            }
            h = next;
        }
    }

    fn peak_entries(&self) -> usize {
        self.peak
    }
}

/// Compute kernel value between two samples
#[inline]
fn kernel_value(x1: &[f64], x2: &[f64], kernel: FastKernel) -> f64 {
    match kernel {
        FastKernel::Linear => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum(),
        FastKernel::Rbf { gamma } => {
            let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            (-gamma * sq_dist).exp()
        }
        FastKernel::Polynomial {
            gamma,
            coef0,
            degree,
        } => {
            let dot: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
            (gamma * dot + coef0).powi(degree.min(i32::MAX as u32) as i32)
        }
    }
}

/// Fill a column in parallel from this many missing entries.
const PARALLEL_FILL: usize = 4096;

/// `Q` on the solver's positions: `rows[pos]` is the row at a position.
struct QMatrix<'a> {
    x: &'a [f64],
    p: usize,
    kernel: FastKernel,
    rows: Vec<usize>,
    y: Vec<f64>,
    qd: Vec<f64>,
    cache: KernelCache,
    evaluations: u64,
}

impl QMatrix<'_> {
    fn row(&self, r: usize) -> &[f64] {
        &self.x[r * self.p..(r + 1) * self.p]
    }

    /// Make sure of `len` entries of column `i`.
    fn get_q(&mut self, i: usize, len: usize) -> EconResult<()> {
        let (start, data) = self.cache.get_data(i, len)?;
        if start >= len {
            return Ok(());
        }
        self.evaluations += (len - start) as u64;
        let (x, p, kernel, rows, y) = (self.x, self.p, self.kernel, &self.rows, &self.y);
        let row_i = &x[rows[i] * p..(rows[i] + 1) * p];
        let yi = y[i];
        let fill = |(offset, slot): (usize, &mut f32)| {
            let j = start + offset;
            let row_j = &x[rows[j] * p..(rows[j] + 1) * p];
            *slot = (yi * y[j] * kernel_value(row_i, row_j, kernel)) as f32;
        };
        if len - start >= PARALLEL_FILL {
            data[start..].par_iter_mut().enumerate().for_each(fill);
        } else {
            data[start..].iter_mut().enumerate().for_each(fill);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Lower,
    Upper,
    Free,
}

const TAU: f64 = 1e-12;

/// LIBSVM's `Solver`, for C-SVC (`p = -1`, one `C`).
struct Solver<'a> {
    q: QMatrix<'a>,
    c: f64,
    eps: f64,
    floor: usize,
    l: usize,
    active: usize,
    alpha: Vec<f64>,
    status: Vec<Bound>,
    g: Vec<f64>,
    g_bar: Vec<f64>,
    unshrink: bool,
    gap: f64,
    min_active: usize,
}

struct Solved {
    iterations: usize,
    converged: bool,
    gap: f64,
    rho: f64,
    objective: f64,
    min_active: usize,
}

impl Solver<'_> {
    fn bound(&self, a: f64) -> Bound {
        if a >= self.c {
            Bound::Upper
        } else if a <= 0.0 {
            Bound::Lower
        } else {
            Bound::Free
        }
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        self.q.cache.swap_index(i, j);
        self.q.rows.swap(i, j);
        self.q.y.swap(i, j);
        self.q.qd.swap(i, j);
        self.alpha.swap(i, j);
        self.status.swap(i, j);
        self.g.swap(i, j);
        self.g_bar.swap(i, j);
    }

    fn reconstruct_gradient(&mut self) -> EconResult<()> {
        let (l, active) = (self.l, self.active);
        if active == l {
            return Ok(());
        }
        for j in active..l {
            self.g[j] = self.g_bar[j] - 1.0;
        }
        let nr_free = self.status[..active]
            .iter()
            .filter(|s| **s == Bound::Free)
            .count();
        if nr_free.saturating_mul(l) > (2 * active).saturating_mul(l - active) {
            for i in active..l {
                resource::checkpoint()?;
                self.q.get_q(i, active)?;
                let qi = self.q.cache.column(i);
                let mut gi = self.g[i];
                for j in 0..active {
                    if self.status[j] == Bound::Free {
                        gi += self.alpha[j] * f64::from(qi[j]);
                    }
                }
                self.g[i] = gi;
            }
        } else {
            for i in 0..active {
                if self.status[i] != Bound::Free {
                    continue;
                }
                resource::checkpoint()?;
                self.q.get_q(i, l)?;
                let qi = self.q.cache.column(i);
                let ai = self.alpha[i];
                for j in active..l {
                    self.g[j] += ai * f64::from(qi[j]);
                }
            }
        }
        Ok(())
    }

    /// WSS3. `None`: `m(a) - M(a) < eps`, or no pair descends.
    fn select_working_set(&mut self) -> EconResult<Option<(usize, usize)>> {
        let active = self.active;
        let (mut gmax, mut gmax2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut gmax_idx = None;
        for t in 0..active {
            let grad = if self.q.y[t] > 0.0 {
                (self.status[t] != Bound::Upper).then_some(-self.g[t])
            } else {
                (self.status[t] != Bound::Lower).then_some(self.g[t])
            };
            if let Some(v) = grad {
                if v >= gmax {
                    gmax = v;
                    gmax_idx = Some(t);
                }
            }
        }
        let Some(i) = gmax_idx else {
            // no variable can go up: the same stopping test, with what is left
            for j in 0..active {
                let v = if self.q.y[j] > 0.0 {
                    (self.status[j] != Bound::Lower).then_some(self.g[j])
                } else {
                    (self.status[j] != Bound::Upper).then_some(-self.g[j])
                };
                if let Some(v) = v {
                    gmax2 = gmax2.max(v);
                }
            }
            self.gap = gmax + gmax2;
            return Ok(None);
        };
        self.q.get_q(i, active)?;
        let qi = self.q.cache.column(i);
        let yi = self.q.y[i];
        let qd_i = self.q.qd[i];
        let mut obj_min = f64::INFINITY;
        let mut gmin_idx = None;
        for j in 0..active {
            let (allowed, signed_g, sign) = if self.q.y[j] > 0.0 {
                (self.status[j] != Bound::Lower, self.g[j], 1.0)
            } else {
                (self.status[j] != Bound::Upper, -self.g[j], -1.0)
            };
            if !allowed {
                continue;
            }
            if signed_g >= gmax2 {
                gmax2 = signed_g;
            }
            let grad_diff = gmax + signed_g;
            if grad_diff > 0.0 {
                let quad = qd_i + self.q.qd[j] - 2.0 * sign * yi * f64::from(qi[j]);
                let denominator = if quad > 0.0 { quad } else { TAU };
                let obj = -(grad_diff * grad_diff) / denominator;
                if obj <= obj_min {
                    gmin_idx = Some(j);
                    obj_min = obj;
                }
            }
        }
        self.gap = gmax + gmax2;
        match gmin_idx {
            Some(j) if self.gap >= self.eps => Ok(Some((i, j))),
            _ => Ok(None),
        }
    }

    fn be_shrunk(&self, i: usize, gmax1: f64, gmax2: f64) -> bool {
        let (y, g) = (self.q.y[i], self.g[i]);
        match self.status[i] {
            Bound::Upper => {
                if y > 0.0 {
                    -g > gmax1
                } else {
                    -g > gmax2
                }
            }
            Bound::Lower => {
                if y > 0.0 {
                    g > gmax2
                } else {
                    g > gmax1
                }
            }
            Bound::Free => false,
        }
    }

    fn do_shrinking(&mut self) -> EconResult<()> {
        let (mut gmax1, mut gmax2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for i in 0..self.active {
            let (y, g, s) = (self.q.y[i], self.g[i], self.status[i]);
            if y > 0.0 {
                if s != Bound::Upper && -g >= gmax1 {
                    gmax1 = -g;
                }
                if s != Bound::Lower && g >= gmax2 {
                    gmax2 = g;
                }
            } else {
                if s != Bound::Upper && -g >= gmax2 {
                    gmax2 = -g;
                }
                if s != Bound::Lower && g >= gmax1 {
                    gmax1 = g;
                }
            }
        }
        if !self.unshrink && gmax1 + gmax2 <= self.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient()?;
            self.active = self.l;
        }
        let floor = self.floor;
        let mut i = 0;
        'scan: while i < self.active {
            if self.be_shrunk(i, gmax1, gmax2) {
                if self.active <= floor {
                    break;
                }
                self.active -= 1;
                while self.active > i {
                    if !self.be_shrunk(self.active, gmax1, gmax2) {
                        self.swap_index(i, self.active);
                        break;
                    }
                    if self.active <= floor {
                        break 'scan;
                    }
                    self.active -= 1;
                }
            }
            i += 1;
        }
        self.min_active = self.min_active.min(self.active);
        Ok(())
    }

    /// One step on the pair (i, j): the two-variable subproblem, then `G` and `G_bar`.
    fn step(&mut self, i: usize, j: usize) -> EconResult<()> {
        self.q.get_q(i, self.active)?;
        self.q.get_q(j, self.active)?;
        let (ai_old, aj_old) = (self.alpha[i], self.alpha[j]);
        let (mut ai, mut aj) = (ai_old, aj_old);
        let c = self.c;
        let qij = f64::from(self.q.cache.column(i)[j]);
        let (qd_i, qd_j) = (self.q.qd[i], self.q.qd[j]);
        if self.q.y[i] != self.q.y[j] {
            let mut quad = qd_i + qd_j + 2.0 * qij;
            if quad <= 0.0 {
                quad = TAU;
            }
            let delta = (-self.g[i] - self.g[j]) / quad;
            let diff = ai - aj;
            ai += delta;
            aj += delta;
            if diff > 0.0 {
                if aj < 0.0 {
                    aj = 0.0;
                    ai = diff;
                }
            } else if ai < 0.0 {
                ai = 0.0;
                aj = -diff;
            }
            if diff > 0.0 {
                if ai > c {
                    ai = c;
                    aj = c - diff;
                }
            } else if aj > c {
                aj = c;
                ai = c + diff;
            }
        } else {
            let mut quad = qd_i + qd_j - 2.0 * qij;
            if quad <= 0.0 {
                quad = TAU;
            }
            let delta = (self.g[i] - self.g[j]) / quad;
            let sum = ai + aj;
            ai -= delta;
            aj += delta;
            if sum > c {
                if ai > c {
                    ai = c;
                    aj = sum - c;
                }
            } else if aj < 0.0 {
                aj = 0.0;
                ai = sum;
            }
            if sum > c {
                if aj > c {
                    aj = c;
                    ai = sum - c;
                }
            } else if ai < 0.0 {
                ai = 0.0;
                aj = sum;
            }
        }
        self.alpha[i] = ai;
        self.alpha[j] = aj;
        let (dai, daj) = (ai - ai_old, aj - aj_old);
        let (qi, qj) = (self.q.cache.column(i), self.q.cache.column(j));
        for (k, g) in self.g[..self.active].iter_mut().enumerate() {
            *g += f64::from(qi[k]) * dai + f64::from(qj[k]) * daj;
        }
        for (index, new) in [(i, ai), (j, aj)] {
            let was_upper = self.status[index] == Bound::Upper;
            self.status[index] = self.bound(new);
            if was_upper != (self.status[index] == Bound::Upper) {
                self.q.get_q(index, self.l)?;
                let column = self.q.cache.column(index);
                let signed = if was_upper { -c } else { c };
                for (gb, q) in self.g_bar.iter_mut().zip(column) {
                    *gb += signed * f64::from(*q);
                }
            }
        }
        Ok(())
    }

    fn calculate_rho(&self) -> f64 {
        let (mut ub, mut lb, mut sum_free, mut nr_free) =
            (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0usize);
        for i in 0..self.active {
            let y = self.q.y[i];
            let yg = y * self.g[i];
            match self.status[i] {
                Bound::Upper => {
                    if y < 0.0 {
                        ub = ub.min(yg);
                    } else {
                        lb = lb.max(yg);
                    }
                }
                Bound::Lower => {
                    if y > 0.0 {
                        ub = ub.min(yg);
                    } else {
                        lb = lb.max(yg);
                    }
                }
                Bound::Free => {
                    nr_free += 1;
                    sum_free += yg;
                }
            }
        }
        if nr_free > 0 {
            sum_free / nr_free as f64
        } else {
            (ub + lb) / 2.0
        }
    }

    fn solve(&mut self, max_iter: usize, shrinking: bool) -> EconResult<Solved> {
        let l = self.l;
        let mut counter = l.min(1000) + 1;
        let mut iterations = 0;
        let mut converged = false;
        while iterations < max_iter {
            counter -= 1;
            if counter == 0 {
                counter = l.min(1000);
                resource::checkpoint()?;
                if shrinking {
                    self.do_shrinking()?;
                }
            }
            let pair = match self.select_working_set()? {
                Some(pair) => pair,
                None => {
                    self.reconstruct_gradient()?;
                    self.active = l;
                    match self.select_working_set()? {
                        Some(pair) => {
                            counter = 1;
                            pair
                        }
                        None => {
                            converged = true;
                            break;
                        }
                    }
                }
            };
            iterations += 1;
            self.step(pair.0, pair.1)?;
        }
        self.reconstruct_gradient()?;
        self.active = l;
        let rho = self.calculate_rho();
        let mut obj = 0.0;
        for i in 0..l {
            obj += self.alpha[i] * (self.g[i] - 1.0);
        }
        Ok(Solved {
            iterations,
            converged,
            gap: self.gap,
            rho,
            objective: -obj / 2.0,
            min_active: self.min_active,
        })
    }
}

fn validate(
    data: &ArrayView2<f64>,
    labels: &ArrayView1<f64>,
    config: &FastSvmConfig,
) -> Result<(), String> {
    let (n, p) = data.dim();
    if n != labels.len() {
        return Err("Data and labels must have same length".to_string());
    }
    if n < 2 {
        return Err(format!("FastSVM: need at least 2 observations, got {n}"));
    }
    if p == 0 {
        return Err("FastSVM: need at least one predictor".to_string());
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("FastSVM: the data contain non-finite values".to_string());
    }
    if !(config.c.is_finite() && config.c > 0.0) {
        return Err(format!(
            "FastSVM: c must be finite and positive, got {}",
            config.c
        ));
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(format!(
            "FastSVM: tolerance must be finite and positive, got {}",
            config.tolerance
        ));
    }
    if config.max_iter == 0 {
        return Err("FastSVM: max_iter must be at least 1, got 0".to_string());
    }
    if config.cache_size_mb == 0 {
        return Err("FastSVM: cache_size_mb must be at least 1, got 0".to_string());
    }
    let gamma_ok = |gamma: f64| {
        if gamma.is_finite() && gamma > 0.0 {
            Ok(())
        } else {
            Err(format!(
                "FastSVM: gamma must be finite and positive, got {gamma}"
            ))
        }
    };
    match config.kernel {
        FastKernel::Linear => {}
        FastKernel::Rbf { gamma } => gamma_ok(gamma)?,
        FastKernel::Polynomial {
            gamma,
            coef0,
            degree,
        } => {
            gamma_ok(gamma)?;
            if degree == 0 {
                return Err("FastSVM: degree must be at least 1, got 0".to_string());
            }
            if !coef0.is_finite() {
                return Err(format!("FastSVM: coef0 must be finite, got {coef0}"));
            }
        }
    }
    Ok(())
}

/// Train a C-SVC.
///
/// # Errors
///
/// A message when `data` and `labels` differ in length, a parameter is out of range, the data or the
/// labels are non-finite, the labels are not exactly two values, the kernel overflows, or the
/// installed computation budget expires.
pub fn fast_svm(
    data: ArrayView2<f64>,
    labels: ArrayView1<f64>,
    config: &FastSvmConfig,
) -> Result<FastSvmResult, String> {
    validate(&data, &labels, config)?;
    let classes = binary_classes(&labels)?;
    train(
        data,
        &classes.y.to_vec(),
        (classes.negative, classes.positive),
        config,
    )
    .map_err(|e| e.to_string())
}

fn train(
    data: ArrayView2<f64>,
    y: &[f64],
    class_labels: (f64, f64),
    config: &FastSvmConfig,
) -> EconResult<FastSvmResult> {
    let (n, p) = data.dim();
    let len = resource::checked_len(&[n, p], "SVM data")?;
    let mut x: Vec<f64> = resource::try_with_capacity(len, "SVM data")?;
    x.extend(data.iter().copied());

    let size_bytes = config.cache_size_mb.saturating_mul(1 << 20);
    let mut qd = resource::try_vec(n, 0.0, "SVM kernel diagonal")?;
    for (slot, row) in qd.iter_mut().zip(x.chunks_exact(p)) {
        *slot = kernel_value(row, row, config.kernel);
    }
    let overflow = || EconError::InvalidSpecification {
        message: "FastSVM: the kernel overflows on these data; rescale them or lower gamma"
            .to_string(),
    };
    if qd.iter().any(|v| !v.is_finite()) {
        return Err(overflow());
    }
    let mut rows = resource::try_vec(n, 0usize, "SVM positions")?;
    rows.iter_mut().enumerate().for_each(|(i, r)| *r = i);
    let mut solver = Solver {
        q: QMatrix {
            x: &x,
            p,
            kernel: config.kernel,
            rows,
            y: y.to_vec(),
            qd,
            cache: KernelCache::new(n, size_bytes)?,
            evaluations: 0,
        },
        c: config.c,
        eps: config.tolerance,
        floor: config.min_working_set,
        l: n,
        active: n,
        alpha: resource::try_vec(n, 0.0, "SVM dual")?,
        status: resource::try_vec(n, Bound::Lower, "SVM dual")?,
        g: resource::try_vec(n, -1.0, "SVM gradient")?,
        g_bar: resource::try_vec(n, 0.0, "SVM gradient")?,
        unshrink: false,
        gap: f64::INFINITY,
        min_active: n,
    };
    let solved = solver.solve(config.max_iter, config.shrinking)?;
    if !solved.rho.is_finite() {
        return Err(overflow());
    }

    let mut alphas = resource::try_vec(n, 0.0, "SVM dual")?;
    for (pos, &row) in solver.q.rows.iter().enumerate() {
        alphas[row] = solver.alpha[pos];
    }
    let mut support_vector_indices = Vec::new();
    let mut dual_coefficients = Vec::new();
    for (row, &a) in alphas.iter().enumerate() {
        if a != 0.0 {
            support_vector_indices.push(row);
            dual_coefficients.push(a * y[row]);
        }
    }

    // Decision values in double precision on every row, as svm_predict_values.
    let bias = -solved.rho;
    let kernel = config.kernel;
    let budget = resource::current_budget();
    let mut decision_values = resource::try_vec(n, 0.0, "SVM decision values")?;
    let (x_ref, svs, coefs) = (&x, &support_vector_indices, &dual_coefficients);
    decision_values
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(r, slot)| -> EconResult<()> {
            if let Some(b) = &budget {
                b.check()?;
            }
            let row = &x_ref[r * p..(r + 1) * p];
            let mut s = 0.0;
            for (&sv, &coef) in svs.iter().zip(coefs) {
                s += coef * kernel_value(row, &x_ref[sv * p..(sv + 1) * p], kernel);
            }
            *slot = s + bias;
            Ok(())
        })?;
    let predictions: Vec<f64> = decision_values
        .iter()
        .map(|&d| {
            if d > 0.0 {
                class_labels.1
            } else {
                class_labels.0
            }
        })
        .collect();

    let convergence = Convergence::new(
        solved.converged,
        solved.iterations,
        config.max_iter,
        "maximal violating pair m(alpha) - M(alpha)",
        solved.gap,
        config.tolerance,
    );
    let warnings: Vec<String> = convergence.note("LIBSVM C-SVC").into_iter().collect();

    Ok(FastSvmResult {
        predictions,
        n_support_vectors: support_vector_indices.len(),
        alphas,
        bias,
        n_iterations: solved.iterations,
        objective_value: solved.objective,
        decision_values,
        support_vector_indices,
        dual_coefficients,
        class_labels,
        converged: solved.converged,
        convergence: Some(convergence),
        warnings,
        kernel_evaluations: solver.q.evaluations,
        cache_peak_entries: solver.q.cache.peak_entries(),
        min_active_size: solved.min_active,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    mod validate {
        use super::super::*;
        use crate::data::columns::{NaPolicy, numeric_column};
        use ndarray::{Array1, Array2};
        use p2a_validation::{Case, Check, Tol};

        pub(super) fn data(label: &str) -> (Case, Array2<f64>, Array1<f64>) {
            let (case, ds) = crate::test_support::load_case("svm_libsvm");
            let col = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
            let (x1, x2) = (col("x1"), col("x2"));
            let x =
                Array2::from_shape_fn((x1.len(), 2), |(i, j)| if j == 0 { x1[i] } else { x2[i] });
            (case, x, Array1::from_vec(col(label)))
        }
        pub(super) fn config(kernel: FastKernel, c: f64, shrinking: bool) -> FastSvmConfig {
            FastSvmConfig {
                kernel,
                c,
                tolerance: 1e-10,
                shrinking,
                cache_size_mb: 40,
                ..Default::default()
            }
        }
        /// 10x the largest measured difference from e1071 in the five runs (validation/ml/svm.md).
        pub(super) const REL: f64 = 4e-7;

        fn gram(x: &Array2<f64>, kernel: FastKernel) -> Array2<f64> {
            let n = x.nrows();
            Array2::from_shape_fn((n, n), |(i, j)| {
                kernel_value(
                    x.row(i).as_slice().unwrap(),
                    x.row(j).as_slice().unwrap(),
                    kernel,
                )
            })
        }
        /// `sum a - 1/2 a' Q a` for signed coefficients `a_i y_i`.
        fn dual_objective(k: &Array2<f64>, index: &[usize], coef: &[f64]) -> f64 {
            let mut quad = 0.0;
            for (&i, &a) in index.iter().zip(coef) {
                for (&j, &b) in index.iter().zip(coef) {
                    quad += a * b * k[[i, j]];
                }
            }
            coef.iter().map(|c| c.abs()).sum::<f64>() - 0.5 * quad
        }
        /// Against e1071::svm, and the properties of a solution of the dual. Before: labels 1 and 2
        /// both mapped to +1, so the dual had no support vector (ML-103).
        pub(super) fn check(
            tag: &str,
            kernel: FastKernel,
            c: f64,
            shrinking: bool,
        ) -> FastSvmResult {
            let linear = matches!(kernel, FastKernel::Linear);
            let (case, x, labels) = data(if linear { "cls_lin" } else { "cls" });
            let r = fast_svm(x.view(), labels.view(), &config(kernel, c, shrinking)).unwrap();
            let tol = Tol::iterative(REL);
            let mut ch = Check::new(&case);
            ch.vector(&format!("{tag}_decision"), &r.decision_values, tol)
                .scalar(&format!("{tag}_bias"), r.bias, tol)
                .vector(&format!("{tag}_pred"), &r.predictions, Tol::exact());
            if linear {
                // the dual need not be unique: only w is compared
                let w: Vec<f64> = (0..2)
                    .map(|j| {
                        r.support_vector_indices
                            .iter()
                            .zip(&r.dual_coefficients)
                            .map(|(&i, &a)| a * x[[i, j]])
                            .sum()
                    })
                    .collect();
                ch.vector(&format!("{tag}_w"), &w, tol);
            } else {
                let sv1: Vec<f64> = r
                    .support_vector_indices
                    .iter()
                    .map(|&i| (i + 1) as f64)
                    .collect();
                ch.scalar(
                    &format!("{tag}_n_sv"),
                    r.n_support_vectors as f64,
                    Tol::exact(),
                )
                .vector(&format!("{tag}_sv_index"), &sv1, Tol::exact())
                .vector(&format!("{tag}_dual_coef"), &r.dual_coefficients, tol);
            }
            ch.finish();
            assert!(r.converged && r.warnings.is_empty(), "{:?}", r.warnings);
            assert_eq!(r.class_labels, (1.0, 2.0));

            // Properties of the solution, on the exact kernel.
            let k = gram(&x, kernel);
            let n = labels.len();
            let y: Vec<f64> = labels
                .iter()
                .map(|&l| if l == 2.0 { 1.0 } else { -1.0 })
                .collect();
            assert!(r.alphas.iter().all(|&a| (0.0..=c).contains(&a)));
            let sum: f64 = (0..n).map(|i| r.alphas[i] * y[i]).sum();
            assert!(sum.abs() <= 1e-12, "sum alpha y = {sum}");
            let (mut up, mut low) = (f64::NEG_INFINITY, f64::INFINITY);
            for i in 0..n {
                let g: f64 = y[i]
                    * r.support_vector_indices
                        .iter()
                        .zip(&r.dual_coefficients)
                        .map(|(&j, &a)| a * k[[i, j]])
                        .sum::<f64>()
                    - 1.0;
                let v = -y[i] * g;
                if (y[i] > 0.0 && r.alphas[i] < c) || (y[i] < 0.0 && r.alphas[i] > 0.0) {
                    up = up.max(v);
                }
                if (y[i] > 0.0 && r.alphas[i] > 0.0) || (y[i] < 0.0 && r.alphas[i] < c) {
                    low = low.min(v);
                }
            }
            // The solver's Q is rounded to single precision (P1), so on the exact kernel the
            // violation reaches its rounding, about 1e-7 * sum alpha, above the 1e-10 it stopped at.
            let rounding = 2e-7 * r.alphas.iter().sum::<f64>().max(1.0);
            assert!(up - low <= 1e-10 + rounding, "KKT violation {}", up - low);
            let index: Vec<usize> = case
                .vector(&format!("{tag}_sv_index"))
                .iter()
                .map(|&i| i as usize - 1)
                .collect();
            let theirs = dual_objective(&k, &index, &case.vector(&format!("{tag}_dual_coef")));
            let ours = dual_objective(&k, &r.support_vector_indices, &r.dual_coefficients);
            assert!(ours >= theirs - 1e-9 * theirs.abs(), "{ours} {theirs}");
            let reported = (r.objective_value - ours).abs();
            assert!(
                reported <= 1e-5 * ours.abs(),
                "{} {ours}",
                r.objective_value
            );
            r
        }
    }

    #[test]
    fn test_validate_fast_svm_libsvm_radial_c01() {
        validate::check("radial_c01", FastKernel::Rbf { gamma: 0.5 }, 0.1, true);
    }
    /// Shrinking does not change the solution.
    #[test]
    fn test_validate_fast_svm_libsvm_radial_c10() {
        let a = validate::check("radial_c10", FastKernel::Rbf { gamma: 0.5 }, 10.0, true);
        let b = validate::check(
            "radial_c10_noshrink",
            FastKernel::Rbf { gamma: 0.5 },
            10.0,
            false,
        );
        assert!(a.min_active_size < 160 && b.min_active_size == 160);
        for (p, q) in a.decision_values.iter().zip(&b.decision_values) {
            assert!(
                (p - q).abs() <= validate::REL * q.abs().max(1e-3),
                "{p} {q}"
            );
        }
    }
    #[test]
    fn test_validate_fast_svm_libsvm_linear_c01() {
        validate::check("linear_c01", FastKernel::Linear, 0.1, true);
    }
    #[test]
    fn test_validate_fast_svm_libsvm_linear_c10() {
        validate::check("linear_c10", FastKernel::Linear, 10.0, true);
    }

    /// ML-103: any two-valued coding gives the same machine; outputs come back in the caller's
    /// labels. The larger label is positive, so a reversed coding negates the decision values.
    #[test]
    fn test_fast_svm_labels_are_mapped_and_returned() {
        let (_, x, cls) = validate::data("cls");
        let cfg = validate::config(FastKernel::Rbf { gamma: 0.5 }, 1.0, true);
        let base = fast_svm(x.view(), cls.view(), &cfg).unwrap();
        for (lo, hi) in [(0.0, 1.0), (-1.0, 1.0), (3.0, 17.0), (0.5, 1.5)] {
            let l = cls.mapv(|v| if v == 1.0 { lo } else { hi });
            let r = fast_svm(x.view(), l.view(), &cfg).unwrap();
            assert_eq!(r.decision_values, base.decision_values, "{lo}/{hi}");
            assert_eq!(r.class_labels, (lo, hi));
            assert!(
                r.predictions
                    .iter()
                    .zip(&base.predictions)
                    .all(|(p, q)| *p == if *q == 1.0 { lo } else { hi })
            );
        }
        let flipped = cls.mapv(|v| if v == 1.0 { 9.0 } else { 4.0 });
        let r = fast_svm(x.view(), flipped.view(), &cfg).unwrap();
        assert_eq!(r.class_labels, (4.0, 9.0));
        for (a, b) in r.decision_values.iter().zip(&base.decision_values) {
            assert!(
                (a + b).abs() <= validate::REL * b.abs().max(1e-3),
                "{a} {b}"
            );
        }
        let one = Array1::from_elem(cls.len(), 1.0);
        assert!(
            fast_svm(x.view(), one.view(), &cfg)
                .unwrap_err()
                .contains("exactly 2 classes")
        );
    }

    /// LIBSVM's cache: least recently used columns are evicted; the budget is max(bytes/4 - 8 l,
    /// 2 l) entries.
    #[test]
    fn test_kernel_cache_evicts_least_recently_used() {
        let mut c = KernelCache::new(10, 0).unwrap(); // budget = 2 l = 20 entries
        assert_eq!(c.get_data(0, 10).unwrap().0, 0); // start 0: column 0 must be filled
        assert_eq!(c.get_data(1, 10).unwrap().0, 0);
        assert_eq!(c.get_data(2, 10).unwrap().0, 0); // evicts column 0 (LRU)
        assert_eq!(c.get_data(1, 10).unwrap().0, 10); // cached: nothing to fill
        assert_eq!(c.get_data(0, 10).unwrap().0, 0); // was evicted; now evicts column 2
        assert_eq!(c.get_data(2, 10).unwrap().0, 0); // evicts column 1
        assert_eq!(c.get_data(0, 6).unwrap().0, 6); // shorter request of a cached column
        assert!(c.peak_entries() <= 20);

        // A swap moves a column and its entries with it; one ending between the two is given up.
        let mut c = KernelCache::new(4, 1 << 20).unwrap();
        c.get_data(0, 4)
            .unwrap()
            .1
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        c.get_data(1, 2).unwrap().1.copy_from_slice(&[5.0, 6.0]);
        c.swap_index(0, 3);
        assert_eq!(c.get_data(3, 4).unwrap().0, 4);
        assert_eq!(c.column(3), &[4.0, 2.0, 3.0, 1.0]);
        // Column 1 ended between the two swapped indices, so it was given up.
        assert_eq!(c.get_data(1, 2).unwrap().0, 0);
    }

    /// ML-103: the kernel cache is bounded by cache_size_mb and never changes the solution.
    /// Before: the full kernel was always materialised.
    #[test]
    fn test_fast_svm_kernel_cache_is_bounded() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(45);
        let n = 1200;
        let x = Array2::from_shape_fn((n, 3), |_| rng.gen_range(-2.0..2.0));
        let y = Array1::from_shape_fn(n, |i| {
            if x[[i, 0]] * x[[i, 1]] + 0.3 * rng.gen_range(-1.0..1.0) > 0.0 {
                1.0
            } else {
                0.0
            }
        });
        let mut small = validate::config(FastKernel::Rbf { gamma: 0.5 }, 1.0, true);
        small.tolerance = 1e-3;
        small.cache_size_mb = 1;
        let mut big = small.clone();
        big.cache_size_mb = 100;
        let (a, b) = (
            fast_svm(x.view(), y.view(), &small).unwrap(),
            fast_svm(x.view(), y.view(), &big).unwrap(),
        );
        assert_eq!(a.decision_values, b.decision_values);
        assert_eq!(a.n_iterations, b.n_iterations);
        assert!(
            a.cache_peak_entries <= (1 << 20) / 4 - 8 * n,
            "{}",
            a.cache_peak_entries
        );
        assert!(a.kernel_evaluations >= b.kernel_evaluations);
    }

    /// min_working_set >= n keeps every variable active: identical to shrinking = false.
    #[test]
    fn test_fast_svm_min_working_set_floor() {
        let (_, x, cls) = validate::data("cls");
        let mut floor = validate::config(FastKernel::Rbf { gamma: 0.5 }, 10.0, true);
        floor.min_working_set = 160;
        let off = validate::config(FastKernel::Rbf { gamma: 0.5 }, 10.0, false);
        let (a, b) = (
            fast_svm(x.view(), cls.view(), &floor).unwrap(),
            fast_svm(x.view(), cls.view(), &off).unwrap(),
        );
        assert_eq!(a.min_active_size, 160);
        assert_eq!(
            (a.n_iterations, &a.decision_values),
            (b.n_iterations, &b.decision_values)
        );
        // An ordinary floor is held too.
        let mut some = validate::config(FastKernel::Rbf { gamma: 0.5 }, 10.0, true);
        some.min_working_set = 80;
        let r = fast_svm(x.view(), cls.view(), &some).unwrap();
        assert!(
            r.converged && r.min_active_size >= 80,
            "{}",
            r.min_active_size
        );
    }

    #[test]
    fn test_fast_svm_reports_iteration_limit() {
        let (_, x, cls) = validate::data("cls");
        let mut c = validate::config(FastKernel::Rbf { gamma: 0.5 }, 10.0, true);
        c.max_iter = 5;
        let r = fast_svm(x.view(), cls.view(), &c).unwrap();
        assert!(!r.converged && r.n_iterations == 5);
        assert!(
            r.warnings.iter().any(|w| w.contains("did not converge")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Converged: no"));
    }

    /// The polynomial kernel solves its dual too; no R case compares it.
    #[test]
    fn test_fast_svm_polynomial_is_feasible() {
        let (_, x, cls) = validate::data("cls");
        let kernel = FastKernel::Polynomial {
            gamma: 0.5,
            coef0: 1.0,
            degree: 2,
        };
        let r = fast_svm(x.view(), cls.view(), &validate::config(kernel, 1.0, true)).unwrap();
        assert!(r.converged);
        let sum: f64 = r.dual_coefficients.iter().sum();
        assert!(sum.abs() <= 1e-12, "{sum}");
        let accuracy = r
            .predictions
            .iter()
            .zip(cls.iter())
            .filter(|(p, c)| p == c)
            .count();
        assert!(accuracy > 120, "{accuracy}"); // the classes are a noisy ring
    }

    #[test]
    fn test_fast_svm_rejects_bad_arguments() {
        let (_, x, cls) = validate::data("cls");
        let bad = |f: &dyn Fn(&mut FastSvmConfig)| {
            let mut c = FastSvmConfig::default();
            f(&mut c);
            fast_svm(x.view(), cls.view(), &c).unwrap_err()
        };
        assert!(bad(&|c| c.c = 0.0).contains("c must"));
        assert!(bad(&|c| c.tolerance = f64::NAN).contains("tolerance"));
        assert!(bad(&|c| c.max_iter = 0).contains("max_iter"));
        assert!(bad(&|c| c.cache_size_mb = 0).contains("cache_size_mb"));
        assert!(bad(&|c| c.kernel = FastKernel::Rbf { gamma: -1.0 }).contains("gamma"));
        assert!(
            bad(&|c| {
                c.kernel = FastKernel::Polynomial {
                    gamma: 1.0,
                    coef0: 0.0,
                    degree: 0,
                }
            })
            .contains("degree")
        );
        let huge = FastKernel::Polynomial {
            gamma: 1e300,
            coef0: 1e300,
            degree: 8,
        };
        assert!(bad(&|c| c.kernel = huge).contains("overflow"));
        let mut xn = x.clone();
        xn[[0, 0]] = f64::INFINITY;
        assert!(
            fast_svm(xn.view(), cls.view(), &FastSvmConfig::default())
                .unwrap_err()
                .contains("non-finite")
        );
        assert!(
            fast_svm(
                x.view(),
                cls.slice(ndarray::s![..10]),
                &FastSvmConfig::default()
            )
            .unwrap_err()
            .contains("same length")
        );
    }
}
