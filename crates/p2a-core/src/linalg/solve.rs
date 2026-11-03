//! Rank-revealing least squares and symmetric inverses: the one numerical path of Tier 3.
//!
//! * [`Qr`], [`least_squares`] and [`weighted_least_squares`] follow R's `lm.fit` and
//!   `lm.wfit`: a Householder QR with LINPACK `dqrdc2`'s limited column pivoting (a
//!   column whose remaining norm falls below `tol` times its original norm is moved to
//!   the end, so the later of two collinear columns is the aliased one; `tol` is 1e-7
//!   as in `lm.fit`). Aliased columns get `NaN` coefficients and `NaN` rows and columns
//!   in the unscaled covariance; [`aliasing_message`] words R's warning.
//! * [`sym_inverse`] inverts a symmetric matrix and reports its numerical rank and
//!   reciprocal condition (of the diagonally equilibrated matrix) instead of silently
//!   returning a pseudoinverse; [`SingularPolicy`] decides whether a rank-deficient
//!   matrix is an error or gets a symmetric generalized inverse.
//!
//! Checked against `base::qr`, `stats::lm.fit`, `stats::lm.wfit` and `stats::lm` on the R
//! case `linalg_qr_glm` (`validation/linalg/solve.md`).
//!
//! ```
//! use ndarray::array;
//! use p2a_core::linalg::solve::least_squares;
//! let x = array![[1.0, 1.0, 2.0], [1.0, 2.0, 4.0], [1.0, 3.0, 6.0], [1.0, 4.0, 8.0]];
//! let y = array![1.1, 1.9, 3.2, 3.8];
//! let fit = least_squares(x.view(), y.view()).unwrap();
//! assert_eq!(fit.rank, 2);
//! assert_eq!(fit.aliased, vec![2]);          // the third column is twice the second
//! assert!(fit.coefficients[2].is_nan());
//! ```

use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{
    CONDITION_THRESHOLD, LinalgError, chol_solve_identity, lu_inverse_bounded, norm1,
};
use crate::resource;
use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// stats::lm.fit tolerance.
pub const LM_QR_TOL: f64 = 1e-7;
/// Relative eigenvalue tolerance of `sym_inverse` on the diagonally equilibrated matrix:
/// eigenvalues <= SYM_RANK_TOL * dim * lambda_max count as zero.
pub const SYM_RANK_TOL: f64 = 1e-14;
/// A matrix is symmetric when max |m_ij - m_ji| <= SYMMETRY_TOL * max |m_ij|.
pub const SYMMETRY_TOL: f64 = 1e-10;

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// Reference-BLAS `dnrm2`: the Euclidean norm with scale/ssq accumulation (no overflow).
fn nrm2(v: &[f64]) -> f64 {
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &x in v {
        if x != 0.0 {
            let a = x.abs();
            if scale < a {
                let r = scale / a;
                ssq = 1.0 + ssq * r * r;
                scale = a;
            } else {
                let r = a / scale;
                ssq += r * r;
            }
        }
    }
    scale * ssq.sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y += t * x`.
fn axpy(t: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += t * xi;
    }
}

fn check_qr_tol(tol: f64) -> EconResult<()> {
    if tol.is_finite() && tol > 0.0 && tol < 1.0 {
        Ok(())
    } else {
        Err(spec(format!("QR tolerance must be in (0, 1), got {tol}")))
    }
}

/// A Householder QR decomposition with `dqrdc2`'s limited column pivoting (R's `qr(x,
/// LAPACK = FALSE)`, the `lm.fit` path).
///
/// The factor is held column by column (the working copy is `X'`), so every norm, dot
/// product and update runs on contiguous memory. The arithmetic follows `dqrdc2`
/// (overflow-safe norms, the norm downdate with Bates' recomputation, the stable
/// Householder sign), but bit-level agreement with `base::qr` is neither required nor
/// tested.
#[derive(Debug, Clone)]
pub struct Qr {
    /// Column `j` of the compact factor at `a[j * n..(j + 1) * n]`: R in the upper
    /// triangle, the Householder vectors (without their first entry) below it.
    a: Vec<f64>,
    n: usize,
    p: usize,
    qraux: Array1<f64>,
    pivot: Vec<usize>,
    rank: usize,
}

impl Qr {
    /// Decompose `x` (n x p) with column tolerance `tol` (0 < tol < 1; R's `lm.fit` uses
    /// [`LM_QR_TOL`]). Refuses a non-finite entry or tolerance.
    pub fn decompose(x: ArrayView2<f64>, tol: f64) -> EconResult<Qr> {
        check_qr_tol(tol)?;
        let (n, p) = x.dim();
        let len = resource::checked_len(&[p, n], "QR workspace")?;
        let mut a = resource::try_vec(len, 0.0, "QR workspace")?;
        for (j, col) in x.columns().into_iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                if !v.is_finite() {
                    return Err(spec(format!(
                        "x contains a non-finite value (row {i}, column {j})"
                    )));
                }
                a[j * n + i] = v;
            }
        }
        Self::factor(a, n, p, tol)
    }

    /// Decompose a matrix given column by column (`cols[j * n + i]` = `x[i, j]`), as the
    /// IRLS loop of `econometrics::glm` builds it.
    pub(crate) fn from_columns(cols: Vec<f64>, n: usize, p: usize, tol: f64) -> EconResult<Qr> {
        check_qr_tol(tol)?;
        if Some(cols.len()) != n.checked_mul(p) {
            return Err(EconError::Computation(format!(
                "QR workspace has {} values for {n} x {p}",
                cols.len()
            )));
        }
        if let Some(k) = cols.iter().position(|v| !v.is_finite()) {
            // a value exists, so n * p > 0 and n > 0
            let (j, i) = (k / n, k % n);
            return Err(spec(format!(
                "x contains a non-finite value (row {i}, column {j})"
            )));
        }
        Self::factor(cols, n, p, tol)
    }

    fn factor(mut a: Vec<f64>, n: usize, p: usize, tol: f64) -> EconResult<Qr> {
        let mut qraux = resource::try_vec(p, 0.0, "QR workspace")?;
        // work2: the original column norms (1 for a zero column), the reference of the
        // negligibility test.
        let mut work2 = resource::try_vec(p, 0.0, "QR workspace")?;
        for j in 0..p {
            qraux[j] = nrm2(&a[j * n..(j + 1) * n]);
            work2[j] = if qraux[j] == 0.0 { 1.0 } else { qraux[j] };
        }
        let mut pivot: Vec<usize> = (0..p).collect();
        let mut k0 = p;
        let lup = n.min(p);
        for l in 0..lup {
            resource::checkpoint()?;
            // A column whose remaining norm is negligible against its original norm is
            // cycled to the end (limited pivoting).
            while l < k0 && qraux[l] < work2[l] * tol {
                a[l * n..p * n].rotate_left(n);
                pivot[l..p].rotate_left(1);
                qraux[l..p].rotate_left(1);
                work2[l..p].rotate_left(1);
                k0 -= 1;
            }
            if l + 1 == n {
                break;
            }
            let (head, tail) = a.split_at_mut((l + 1) * n);
            let cl = &mut head[l * n..];
            let mut nrmxl = nrm2(&cl[l..]);
            if nrmxl == 0.0 {
                // No reflector for this column. dqrdc2 leaves qraux[l] at the downdated
                // norm here; zero marks "no reflector" for qty and q_times (dqrsl skips
                // a reflector with qraux 0).
                qraux[l] = 0.0;
                continue;
            }
            if cl[l] != 0.0 {
                nrmxl = nrmxl.abs().copysign(cl[l]);
            }
            let r = 1.0 / nrmxl;
            for v in &mut cl[l..] {
                *v *= r;
            }
            cl[l] += 1.0;
            for (jj, cj) in tail.chunks_exact_mut(n).enumerate() {
                let j = l + 1 + jj;
                let t = -dot(&cl[l..], &cj[l..]) / cl[l];
                axpy(t, &cl[l..], &mut cj[l..]);
                if qraux[j] != 0.0 {
                    let ratio = cj[l].abs() / qraux[j];
                    let tt = (1.0 - ratio * ratio).max(0.0);
                    if tt.abs() < 1e-6 {
                        // Bates (1999): recompute instead of downdating a norm that has
                        // lost its precision.
                        qraux[j] = nrm2(&cj[l + 1..]);
                    } else {
                        qraux[j] *= tt.sqrt();
                    }
                }
            }
            qraux[l] = cl[l];
            cl[l] = -nrmxl;
        }
        Ok(Qr {
            a,
            n,
            p,
            qraux: Array1::from_vec(qraux),
            pivot,
            rank: k0.min(n),
        })
    }

    /// Column `j` of the compact factor.
    fn col(&self, j: usize) -> &[f64] {
        &self.a[j * self.n..(j + 1) * self.n]
    }

    /// `R[i, j]` (`i <= j`).
    fn r(&self, i: usize, j: usize) -> f64 {
        self.a[j * self.n + i]
    }

    /// Numerical rank (R's `qr$rank`).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Column order of the factor, 0-based (R's `qr$pivot - 1`): `pivot()[k]` is the
    /// original index of the k-th column of R.
    pub fn pivot(&self) -> &[usize] {
        &self.pivot
    }

    /// LINPACK's compact storage of the reflectors' first entries (R's `qr$qraux`).
    pub(crate) fn qraux(&self) -> ArrayView1<'_, f64> {
        self.qraux.view()
    }

    /// Original indices of the aliased columns (`pivot()[rank..]`), ascending.
    pub fn aliased(&self) -> Vec<usize> {
        let mut v = self.pivot[self.rank..].to_vec();
        v.sort_unstable();
        v
    }

    /// Number of reflectors that act (dqrsl's `ju`).
    fn n_reflectors(&self) -> usize {
        self.rank.min(self.n.saturating_sub(1))
    }

    /// Apply reflector `j` to `q` in place.
    fn reflect(&self, j: usize, q: &mut [f64]) {
        let v0 = self.qraux[j];
        if v0 == 0.0 {
            return;
        }
        let c = self.col(j);
        let t = -(v0 * q[j] + dot(&c[j + 1..], &q[j + 1..])) / v0;
        q[j] += t * v0;
        axpy(t, &c[j + 1..], &mut q[j + 1..]);
    }

    /// [`EconError::InvalidSpecification`] unless `y` has one value per row of the
    /// decomposed matrix.
    fn check_len(&self, y: &ArrayView1<f64>) -> EconResult<()> {
        if y.len() != self.n {
            return Err(spec(format!(
                "y has {} rows but the decomposed matrix has {}",
                y.len(),
                self.n
            )));
        }
        Ok(())
    }

    /// `Q' y` (dqrsl job 1000), with the first `rank` reflectors. Refuses a `y` whose
    /// length is not the number of rows of the decomposed matrix.
    pub fn qty(&self, y: ArrayView1<f64>) -> EconResult<Array1<f64>> {
        self.check_len(&y)?;
        let mut q = y.to_vec();
        for j in 0..self.n_reflectors() {
            self.reflect(j, &mut q);
        }
        Ok(Array1::from_vec(q))
    }

    /// `Q v` with the first `rank` reflectors (the inverse of [`Qr::qty`]). Refuses a `v`
    /// whose length is not the number of rows of the decomposed matrix, as [`Qr::qty`].
    pub(crate) fn q_times(&self, v: ArrayView1<f64>) -> EconResult<Array1<f64>> {
        self.check_len(&v)?;
        let mut q = v.to_vec();
        for j in (0..self.n_reflectors()).rev() {
            self.reflect(j, &mut q);
        }
        Ok(Array1::from_vec(q))
    }

    /// Solve `R11 b = (Q'y)[..rank]` and return `b` in pivoted order.
    fn solve_r11(&self, y: ArrayView1<f64>) -> EconResult<Vec<f64>> {
        let q = self.qty(y)?;
        let k = self.rank;
        let mut b = q.to_vec();
        b.truncate(k);
        for j in (0..k).rev() {
            b[j] /= self.r(j, j);
            let bj = b[j];
            axpy(-bj, &self.col(j)[..j], &mut b[..j]);
        }
        Ok(b)
    }

    /// Least-squares coefficients in the original column order, `NaN` for aliased
    /// columns (`lm.fit`). Refuses a `y` of the wrong length, as [`Qr::qty`].
    pub fn coefficients(&self, y: ArrayView1<f64>) -> EconResult<Array1<f64>> {
        let b = self.solve_r11(y)?;
        let mut out = Array1::from_elem(self.p, f64::NAN);
        for (i, v) in b.into_iter().enumerate() {
            out[self.pivot[i]] = v;
        }
        Ok(out)
    }

    /// As [`Qr::coefficients`] with 0 for aliased columns (`dqrls`, used by IRLS).
    pub(crate) fn coefficients_zeroed(&self, y: ArrayView1<f64>) -> EconResult<Array1<f64>> {
        let b = self.solve_r11(y)?;
        let mut out = Array1::zeros(self.p);
        for (i, v) in b.into_iter().enumerate() {
            out[self.pivot[i]] = v;
        }
        Ok(out)
    }

    /// `(R11' R11)^{-1}` (R's `chol2inv(qr$qr[1:rank, 1:rank])`) in the original column
    /// order, with `NaN` rows and columns for aliased columns. Both buffers (`rank^2` and
    /// `p^2`) go through the allocation ceiling: `p` can exceed the rows `decompose`
    /// checked.
    pub fn unscaled_covariance(&self) -> EconResult<Array2<f64>> {
        let k = self.rank;
        // R11^{-1}, upper triangular, column by column: rinv[j * k + i].
        let len = resource::checked_len(&[k, k], "unscaled covariance")?;
        let mut rinv = resource::try_vec(len, 0.0, "unscaled covariance")?;
        for c in 0..k {
            let col = &mut rinv[c * k..(c + 1) * k];
            col[c] = 1.0;
            for j in (0..=c).rev() {
                col[j] /= self.r(j, j);
                let bj = col[j];
                axpy(-bj, &self.col(j)[..j], &mut col[..j]);
            }
        }
        let mut out = resource::try_zeros2(self.p, self.p, "unscaled covariance")?;
        out.fill(f64::NAN);
        for i in 0..k {
            for j in 0..=i {
                // (Rinv Rinv')_{ij} = sum_l Rinv[i, l] Rinv[j, l], l >= max(i, j) = i
                let mut s = 0.0;
                for l in i..k {
                    s += rinv[l * k + i] * rinv[l * k + j];
                }
                out[[self.pivot[i], self.pivot[j]]] = s;
                out[[self.pivot[j], self.pivot[i]]] = s;
            }
        }
        Ok(out)
    }

    /// `max |R_ii| / min |R_ii|` over `i < rank`; `NaN` when the rank is 0.
    pub fn r_diag_ratio(&self) -> f64 {
        if self.rank == 0 {
            return f64::NAN;
        }
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for i in 0..self.rank {
            let v = self.r(i, i).abs();
            lo = lo.min(v);
            hi = hi.max(v);
        }
        hi / lo
    }

    /// The diagonal of the compact factor (R's `diag(qr$qr)`), in pivoted order, length
    /// `min(n, p)`. Entries past the rank belong to aliased columns.
    pub fn r_diag(&self) -> Array1<f64> {
        Array1::from_iter((0..self.n.min(self.p)).map(|i| self.r(i, i)))
    }
}

/// A least-squares fit (`lm.fit` / `lm.wfit`).
#[derive(Debug, Clone)]
pub struct LeastSquares {
    /// Coefficients in the original column order; `NaN` for aliased columns.
    pub coefficients: Array1<f64>,
    /// `y - residuals`.
    pub fitted: Array1<f64>,
    /// Residuals (on the original scale for a weighted fit).
    pub residuals: Array1<f64>,
    /// Numerical rank.
    pub rank: usize,
    /// Original indices of the aliased columns, ascending.
    pub aliased: Vec<usize>,
    /// Residual degrees of freedom: rows (with positive weight) minus rank.
    pub df_residual: usize,
    /// `(X'X)^{-1}` (weighted: `(X'WX)^{-1}`) in the original order, `NaN` rows and
    /// columns for aliased columns. Multiply by the residual variance for `vcov`.
    pub unscaled_cov: Array2<f64>,
    /// The decomposition (of `sqrt(w) X` on the positive-weight rows for a weighted fit).
    pub qr: Qr,
}

impl LeastSquares {
    /// R's "Coefficients not defined because of singularities" message; see
    /// [`aliasing_message`].
    pub fn aliasing_warning(&self, names: &[String]) -> Option<String> {
        aliasing_message(&self.aliased, names)
    }
}

/// `least_squares_tol(x, y, LM_QR_TOL)`: `lm.fit`.
pub fn least_squares(x: ArrayView2<f64>, y: ArrayView1<f64>) -> EconResult<LeastSquares> {
    least_squares_tol(x, y, LM_QR_TOL)
}

/// Least squares by pivoted QR with column tolerance `tol` (`lm.fit(x, y, tol)`).
/// Refuses a length mismatch, non-finite input, an invalid tolerance and zero rows.
pub fn least_squares_tol(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    tol: f64,
) -> EconResult<LeastSquares> {
    let n = x.nrows();
    if y.len() != n {
        return Err(spec(format!("y has {} rows but x has {n}", y.len())));
    }
    if n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "least squares".into(),
        });
    }
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(spec(format!("y contains a non-finite value (row {i})")));
    }
    let qr = Qr::decompose(x, tol)?;
    fit_from_qr(qr, y)
}

fn fit_from_qr(qr: Qr, y: ArrayView1<f64>) -> EconResult<LeastSquares> {
    let n = y.len();
    let rank = qr.rank();
    let mut effects = qr.qty(y)?;
    effects.iter_mut().take(rank).for_each(|v| *v = 0.0);
    let residuals = qr.q_times(effects.view())?;
    let fitted = &y - &residuals;
    let coefficients = qr.coefficients(y)?;
    let unscaled_cov = qr.unscaled_covariance()?;
    Ok(LeastSquares {
        coefficients,
        fitted,
        residuals,
        rank,
        aliased: qr.aliased(),
        df_residual: n - rank,
        unscaled_cov,
        qr,
    })
}

/// Weighted least squares (`lm.wfit`): rows with weight 0 are dropped from the fit (and
/// from the residual degrees of freedom); their fitted value is `x0 b` (aliased
/// coefficients count as 0) and their residual `y0 - x0 b`. Residuals of the other rows
/// are on the original scale. Refuses negative or non-finite weights and an all-zero
/// weight vector (where `lm.wfit` returns an all-`NA` fit).
pub fn weighted_least_squares(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    w: ArrayView1<f64>,
) -> EconResult<LeastSquares> {
    let (n, p) = x.dim();
    if y.len() != n {
        return Err(spec(format!("y has {} rows but x has {n}", y.len())));
    }
    if w.len() != n {
        return Err(spec(format!("weights has {} rows but x has {n}", w.len())));
    }
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(spec(format!("y contains a non-finite value (row {i})")));
    }
    // Every row, as lm.wfit: a zero-weight row's fitted value is x0 b.
    if let Some(((i, j), _)) = x.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(spec(format!(
            "x contains a non-finite value (row {i}, column {j})"
        )));
    }
    if let Some((i, v)) = w
        .iter()
        .enumerate()
        .find(|(_, v)| !v.is_finite() || **v < 0.0)
    {
        return Err(spec(format!(
            "weights must be finite and >= 0 (row {i} is {v})"
        )));
    }
    let ok: Vec<usize> = (0..n).filter(|&i| w[i] > 0.0).collect();
    if ok.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "weighted least squares: no positive weights".into(),
        });
    }
    let m = ok.len();
    let mut xs = resource::try_zeros2(m, p, "weighted least squares")?;
    let mut ys = resource::try_zeros1(m, "weighted least squares")?;
    let mut sw = resource::try_zeros1(m, "weighted least squares")?;
    for (k, &i) in ok.iter().enumerate() {
        let s = w[i].sqrt();
        sw[k] = s;
        ys[k] = y[i] * s;
        let mut finite = ys[k].is_finite();
        for j in 0..p {
            xs[[k, j]] = x[[i, j]] * s;
            finite &= xs[[k, j]].is_finite();
        }
        // A finite but very large weight can still overflow the scaled row; name the
        // caller's row and the weight, not a row of the compacted matrix.
        if !finite {
            return Err(spec(format!(
                "weights: row {i} (w = {:e}) overflows the weighted design",
                w[i]
            )));
        }
    }
    let fit = least_squares_tol(xs.view(), ys.view(), LM_QR_TOL)?;
    let b0 = fit.coefficients.mapv(|v| if v.is_nan() { 0.0 } else { v });
    let mut residuals = Array1::zeros(n);
    let mut fitted = Array1::zeros(n);
    let mut next = ok.iter().copied().enumerate().peekable();
    for i in 0..n {
        if let Some(&(k, _)) = next.peek().filter(|(_, ii)| *ii == i) {
            next.next();
            residuals[i] = fit.residuals[k] / sw[k];
            fitted[i] = y[i] - residuals[i];
        } else {
            let f0 = x.row(i).dot(&b0);
            fitted[i] = f0;
            residuals[i] = y[i] - f0;
        }
    }
    Ok(LeastSquares {
        fitted,
        residuals,
        ..fit
    })
}

/// R's "Coefficients not defined because of singularities: x2, x5", or `None` when
/// nothing is aliased. `names[j]` names column `j` when present; otherwise R's default
/// name `x{j+1}` is used. Shared by [`LeastSquares`] and GLM callers.
pub fn aliasing_message(aliased: &[usize], names: &[String]) -> Option<String> {
    if aliased.is_empty() {
        return None;
    }
    let list: Vec<String> = aliased
        .iter()
        .map(|&j| {
            names
                .get(j)
                .cloned()
                .unwrap_or_else(|| format!("x{}", j + 1))
        })
        .collect();
    Some(format!(
        "Coefficients not defined because of singularities: {}",
        list.join(", ")
    ))
}

/// What [`sym_inverse`] does with a rank-deficient matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingularPolicy {
    /// Refuse it ([`EconError::SingularMatrix`]).
    Error,
    /// Return a symmetric generalized inverse and set [`SymInverse::generalized`].
    Generalized,
}

/// The result of [`sym_inverse`].
#[derive(Debug, Clone)]
pub struct SymInverse {
    /// The inverse, or a symmetric generalized inverse when `generalized`.
    pub inverse: Array2<f64>,
    /// Numerical rank: eigenvalues of the equilibrated matrix above
    /// `SYM_RANK_TOL * dim * lambda_max` in absolute value.
    pub rank: usize,
    /// Dimension of the matrix.
    pub dim: usize,
    /// Reciprocal condition of the diagonally equilibrated matrix `E = S m S`
    /// (`S = diag(1 / sqrt|m_ii|)`, 1 for a zero diagonal entry); 0 for the zero matrix.
    /// It is `min|lambda| / max|lambda|` when that is below 1e-10 or the matrix is not
    /// definite. For a definite matrix that the O(d^2) certificate accepts it is
    /// `1 / kappa_1(E)`, a lower bound within a factor `dim` of the exact value and
    /// always above 1e-10, so a threshold at or below 1e-10 decides exactly.
    pub reciprocal_condition: f64,
    /// `true` when `rank < dim` (a generalized inverse was returned).
    pub generalized: bool,
}

impl SymInverse {
    /// `rank == dim`.
    pub fn is_full_rank(&self) -> bool {
        self.rank == self.dim
    }

    /// "`<what>` is singular (rank r of d); a generalized inverse was used", when
    /// `generalized`.
    pub fn warning(&self, what: &str) -> Option<String> {
        self.generalized.then(|| {
            format!(
                "{what} is singular (rank {} of {}); a generalized inverse was used",
                self.rank, self.dim
            )
        })
    }
}

/// `(max |m_ij - m_ji|, max |m_ij|)` of a square matrix in one pass over its pairs, or
/// `None` when an entry is not finite.
pub(crate) fn finite_asymmetry(m: &ArrayView2<f64>) -> Option<(f64, f64)> {
    let d = m.nrows();
    let Some(sl) = m.as_slice() else {
        // Not row-major contiguous: indexed reads.
        let (mut asym, mut big, mut finite) = (0.0f64, 0.0f64, true);
        for i in 0..d {
            for j in i..d {
                let (a, b) = (m[[i, j]], m[[j, i]]);
                finite &= a.is_finite() & b.is_finite();
                big = big.max(a.abs()).max(b.abs());
                asym = asym.max((a - b).abs());
            }
        }
        return finite.then_some((asym, big));
    };
    // Row-major and contiguous (the usual case). Pass 1, contiguous: `v * 0` is 0 for a
    // finite entry and NaN otherwise, so the sum is NaN exactly when one is not finite.
    // Four lanes keep the reductions out of a single dependency chain.
    let (mut nan, mut big) = ([0.0f64; 4], [0.0f64; 4]);
    let (chunks, rest) = sl.as_chunks::<4>();
    for c in chunks {
        for l in 0..4 {
            nan[l] += c[l] * 0.0;
            let v = c[l].abs();
            if v > big[l] {
                big[l] = v;
            }
        }
    }
    for &v in rest {
        nan[0] += v * 0.0;
        if v.abs() > big[0] {
            big[0] = v.abs();
        }
    }
    if (nan[0] + nan[1] + nan[2] + nan[3]).is_nan() {
        return None;
    }
    let big = big[0].max(big[1]).max(big[2].max(big[3]));
    // Pass 2: the strict lower triangle of row i against column i above the diagonal.
    let mut asym = 0.0f64;
    for i in 1..d {
        for (j, a) in sl[i * d..i * d + i].iter().enumerate() {
            let v = (a - sl[j * d + i]).abs();
            if v > asym {
                asym = v;
            }
        }
    }
    Some((asym, big))
}

/// `m` as a faer matrix (bit for bit), with plain slice reads when `m` is row-major.
fn to_faer(m: &ArrayView2<f64>) -> Mat<f64> {
    let (rows, cols) = m.dim();
    match m.as_slice() {
        Some(sl) => Mat::from_fn(rows, cols, |i, j| sl[i * cols + j]),
        None => Mat::from_fn(rows, cols, |i, j| m[[i, j]]),
    }
}

/// A faer matrix as a row-major ndarray (bit for bit), read through column slices.
fn from_faer(x: &Mat<f64>) -> Array2<f64> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut v = vec![0.0f64; rows * cols];
    for j in 0..cols {
        for (dst, &val) in v[j..].iter_mut().step_by(cols).zip(x.col_as_slice(j)) {
            *dst = val;
        }
    }
    Array2::from_shape_vec((rows, cols), v).expect("rows * cols entries")
}

/// `(rank, reciprocal condition)` from eigenvalues of the equilibrated matrix.
fn classify(lam: &[f64], d: usize) -> (usize, f64, f64) {
    let lmax = lam.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    if lmax == 0.0 {
        return (0, 0.0, 0.0);
    }
    let tol = SYM_RANK_TOL * d as f64 * lmax;
    let rank = lam.iter().filter(|v| v.abs() > tol).count();
    let lmin = lam.iter().fold(f64::INFINITY, |a, v| a.min(v.abs()));
    (rank, lmin / lmax, tol)
}

/// `S (sum over the `keep` eigenpairs of E with the largest |lambda| of v v' / lambda) S`.
/// `guarded`: allocate through the allocation ceiling (see [`sym_inverse_unguarded`]).
fn eigen_inverse(e: &Mat<f64>, s: &[f64], keep: usize, guarded: bool) -> EconResult<Array2<f64>> {
    let d = s.len();
    let eig = e
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|_| LinalgError::EigenFailed)?;
    let lam = eig.S().column_vector();
    let u = eig.U();
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| lam[b].abs().total_cmp(&lam[a].abs()));
    let mut g = if guarded {
        resource::try_zeros2(d, d, "sym_inverse")?
    } else {
        Array2::zeros((d, d))
    };
    for &k in order.iter().take(keep) {
        let inv = 1.0 / lam[k];
        for i in 0..d {
            let ui = u[(i, k)] * inv;
            for j in 0..d {
                g[[i, j]] += ui * u[(j, k)];
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            g[[i, j]] *= s[i] * s[j];
        }
    }
    Ok(g)
}

/// Inverse of a symmetric matrix with its numerical rank.
///
/// * Refused: a non-square matrix (`LinalgError::NotSquare`), a non-finite entry, and an
///   asymmetry above `SYMMETRY_TOL * max |m|` ([`EconError::InvalidSpecification`]). The
///   lower triangle defines the matrix.
/// * Rank and conditioning are those of the equilibrated `E = S m S`,
///   `S = diag(1 / sqrt|m_ii|)` (1 for a zero diagonal entry, so a KKT system stays
///   invertible): eigenvalues at or below `SYM_RANK_TOL * dim * lambda_max` count as zero.
/// * Full rank: positive-definite matrices are inverted by Cholesky (the code
///   `safe_inverse` always ran, bit for bit), negative-definite ones by the Cholesky
///   factor of `-m`, and indefinite ones by the partial-pivot LU of `matrix_ops`,
///   accepted when `max |m X - I|` is within the residual allowance for `kappa_1(m)`
///   (capped by a bound on `kappa_2(m)` from the eigenvalues of `E`; otherwise the
///   eigenvalue inverse). A Cholesky inverse whose `kappa_1(E)` is at most `1e10` (and at most
///   `0.1 / (SYM_RANK_TOL * dim)`) is accepted in O(dim^2) without eigenvalues.
/// * Rank-deficient: [`SingularPolicy::Error`] gives [`EconError::SingularMatrix`];
///   [`SingularPolicy::Generalized`] gives `S E_r^+ S`, the Moore-Penrose inverse of `E`
///   on its identified eigenvalues mapped back, a symmetric reflexive generalized
///   inverse (`m G m = m`, `G m G = G`) that reproduces every estimable quantity.
///
/// `what` names the matrix in messages.
pub fn sym_inverse(
    m: ArrayView2<f64>,
    policy: SingularPolicy,
    what: &str,
) -> EconResult<SymInverse> {
    let (rows, cols) = m.dim();
    if rows != cols {
        return Err(LinalgError::NotSquare { rows, cols }.into());
    }
    let Some((asym, big)) = finite_asymmetry(&m) else {
        return Err(spec(format!("{what} contains non-finite entries")));
    };
    if asym > SYMMETRY_TOL * big {
        return Err(spec(format!(
            "{what} is not symmetric (max |m - m'| = {asym:.3e} > {SYMMETRY_TOL:.0e} * max |m|); \
             sym_inverse requires a symmetric matrix"
        )));
    }
    sym_inverse_core(m, policy, what, true)
}

/// [`sym_inverse`] for `matrix_ops::safe_inverse`, which has already checked that `m` is
/// square, finite and symmetric to [`SYMMETRY_TOL`] (the checks are not repeated), and
/// which has never consulted budgets or the allocation ceiling: its callers already hold
/// the `dim x dim` matrix and run their own `checkpoint()` per iteration, so a
/// cancellation surfaces there as `Cancelled` instead of as a failed inverse. The only
/// error is then `LinalgError::EigenFailed` (and `SingularMatrix` under
/// [`SingularPolicy::Error`]).
pub(crate) fn sym_inverse_unguarded(
    m: ArrayView2<f64>,
    policy: SingularPolicy,
) -> EconResult<SymInverse> {
    debug_assert!(
        m.nrows() == m.ncols()
            && finite_asymmetry(&m).is_some_and(|(asym, big)| asym <= SYMMETRY_TOL * big)
    );
    sym_inverse_core(m, policy, "matrix", false)
}

/// `(||E||_1, ||E^-1||_1)` for `E = S m S`, `S = diag(1 / r)`, from the faer copy `a` of `m`
/// (its lower triangle) and the candidate inverse `x` of `m` (`E^-1 = R x R`); `None` when
/// `x` has a non-finite entry. One pass over each matrix, column by column; `col_e` (length
/// `dim`) is scratch.
fn equilibrated_norms(
    a: &Mat<f64>,
    x: &Mat<f64>,
    r: &[f64],
    s: &[f64],
    col_e: &mut [f64],
) -> Option<(f64, f64)> {
    let d = r.len();
    col_e.fill(0.0);
    let mut einv_norm = 0.0f64;
    for j in 0..d {
        let (sj, rj) = (s[j], r[j]);
        // Column j of E from the lower triangle; each entry below the diagonal also
        // belongs to column i (E is symmetric).
        let lower = &a.col_as_slice(j)[j..];
        col_e[j] += weighted_abs_sum(lower, &s[j..]) * sj;
        for ((v, si), ce) in lower[1..].iter().zip(&s[j + 1..]).zip(&mut col_e[j + 1..]) {
            *ce += v.abs() * si * sj;
        }
        // A non-finite entry of x makes the sum non-finite (the slow check tells it from
        // overflow).
        let col = x.col_as_slice(j);
        let ci = weighted_abs_sum(col, r) * rj;
        if !ci.is_finite() && !col.iter().all(|v| v.is_finite()) {
            return None;
        }
        einv_norm = einv_norm.max(ci);
    }
    let e_norm = col_e.iter().fold(0.0f64, |acc, v| acc.max(*v));
    Some((e_norm, einv_norm))
}

/// `sum |v_i| w_i` over the common length, in four lanes (no single dependency chain).
fn weighted_abs_sum(v: &[f64], w: &[f64]) -> f64 {
    let n = v.len().min(w.len());
    let (v, w) = (&v[..n], &w[..n]);
    let mut acc = [0.0f64; 4];
    let (vc, vr) = v.as_chunks::<4>();
    let (wc, wr) = w.as_chunks::<4>();
    for (v4, w4) in vc.iter().zip(wc) {
        for l in 0..4 {
            acc[l] += v4[l].abs() * w4[l];
        }
    }
    for (vi, wi) in vr.iter().zip(wr) {
        acc[0] += vi.abs() * wi;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// The work of [`sym_inverse`] on a validated (square, finite, symmetric) matrix.
/// `guarded`: run the budget checkpoint and allocate through the allocation ceiling.
fn sym_inverse_core(
    m: ArrayView2<f64>,
    policy: SingularPolicy,
    what: &str,
    guarded: bool,
) -> EconResult<SymInverse> {
    let d = m.nrows();
    if d == 0 {
        return Ok(SymInverse {
            inverse: Array2::zeros((0, 0)),
            rank: 0,
            dim: 0,
            reciprocal_condition: 1.0,
            generalized: false,
        });
    }
    if guarded {
        resource::checkpoint()?;
        let len = resource::checked_len(&[d, d], "sym_inverse")?;
        resource::check_alloc::<f64>(len, "sym_inverse")?;
    }
    let lower = |i: usize, j: usize| if i >= j { m[[i, j]] } else { m[[j, i]] };
    // r_i = sqrt|m_ii| (1 for a zero diagonal entry) and s_i = 1 / r_i: E = S m S.
    // One allocation for r, s and the certificate's column sums.
    let mut scratch = vec![0.0f64; 3 * d];
    let (r, rest) = scratch.split_at_mut(d);
    let (s, col_e) = rest.split_at_mut(d);
    for i in 0..d {
        let a = m[[i, i]].abs();
        r[i] = if a > 0.0 { a.sqrt() } else { 1.0 };
        s[i] = 1.0 / r[i];
    }
    let (r, s) = (&*r, &*s);

    // 1. Cholesky of m, or of -m when every diagonal entry is negative.
    let a = to_faer(&m);
    let mut x = chol_solve_identity(&a);
    if x.is_none() && (0..d).all(|i| m[[i, i]] < 0.0) {
        x = chol_solve_identity(&Mat::from_fn(d, d, |i, j| -a[(i, j)])).map(|mut inv| {
            for j in 0..d {
                for v in inv.col_as_slice_mut(j) {
                    *v = -*v;
                }
            }
            inv
        });
    }

    // 2. Certificate: kappa_2(E) <= kappa_1(E) = ||E||_1 ||E^-1||_1 for a symmetric E. A
    //    non-finite inverse is dropped here.
    if let Some(inv) = &x {
        match equilibrated_norms(&a, inv, r, s, col_e) {
            None => x = None,
            Some((e_norm, einv_norm)) => {
                let kappa1 = e_norm * einv_norm;
                if kappa1 <= CONDITION_THRESHOLD.min(0.1 / (SYM_RANK_TOL * d as f64)) {
                    return Ok(SymInverse {
                        inverse: from_faer(inv),
                        rank: d,
                        dim: d,
                        reciprocal_condition: 1.0 / kappa1,
                        generalized: false,
                    });
                }
            }
        }
    }

    // 3. Eigenvalues of the equilibrated matrix.
    let e = Mat::from_fn(d, d, |i, j| lower(i, j) * s[i] * s[j]);
    let lam = e
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|_| LinalgError::EigenFailed)?;
    let (rank, rcond, _) = classify(&lam, d);
    let full = |inverse: Array2<f64>| SymInverse {
        inverse,
        rank: d,
        dim: d,
        reciprocal_condition: rcond,
        generalized: false,
    };
    if rank == d {
        if let Some(inv) = x {
            return Ok(full(from_faer(&inv)));
        }
        // Indefinite: the verified partial-pivot LU. Its residual allowance uses
        // kappa_1(m) from the candidate, capped by an upper bound on kappa_2(m) that does
        // not depend on it: m^-1 = S E^-1 S gives ||m^-1||_2 <= max s_i^2 / min |lambda(E)|,
        // and ||m||_2 <= ||m||_1 for a symmetric m. (The condition of E alone would make
        // the allowance too tight on a badly scaled matrix.)
        let lmin = lam.iter().fold(f64::INFINITY, |acc, v| acc.min(v.abs()));
        let smax2 = s.iter().fold(0.0f64, |acc, v| acc.max(v * v));
        let cap = norm1(&a) * smax2 / lmin;
        if let Some(inv) = lu_inverse_bounded(&a, cap) {
            return Ok(full(from_faer(&inv)));
        }
        return Ok(full(eigen_inverse(&e, s, d, guarded)?));
    }
    match policy {
        SingularPolicy::Error => Err(EconError::SingularMatrix {
            context: format!("{what}: rank {rank} of {d}"),
            suggestion: "check for perfectly collinear variables (a full set of dummies with \
                         an intercept, or a duplicated, rescaled or constant column)"
                .into(),
        }),
        SingularPolicy::Generalized => Ok(SymInverse {
            inverse: eigen_inverse(&e, s, rank, guarded)?,
            rank,
            dim: d,
            reciprocal_condition: rcond,
            generalized: true,
        }),
    }
}

#[cfg(test)]
pub(crate) mod case_data {
    use crate::data::Dataset;
    use crate::data::columns::{ColumnRequest, Columns, NaPolicy};
    use ndarray::Array2;

    pub(crate) const COLUMNS: &[&str] = &[
        "x1", "x2", "x3", "x4", "off", "yb", "yp", "wq", "yq", "ysep", "yc", "w0",
    ];

    /// All columns of linalg_glm.csv (no missing values; 120 rows).
    pub(crate) fn columns(ds: &Dataset) -> Columns {
        let c = ColumnRequest::new(ds.df())
            .numerics(COLUMNS)
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        assert_eq!(c.n(), 120);
        c
    }

    /// Design matrix with the named columns in order; "1" is a column of ones, "1-<name>"
    /// is 1 - column.
    pub(crate) fn design(c: &Columns, names: &[&str]) -> Array2<f64> {
        let n = c.n();
        let mut x = Array2::<f64>::zeros((n, names.len()));
        for (j, name) in names.iter().enumerate() {
            for i in 0..n {
                x[[i, j]] = match *name {
                    "1" => 1.0,
                    s if s.starts_with("1-") => 1.0 - c.numeric(&s[2..])[i],
                    s => c.numeric(s)[i],
                };
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::case_data::{columns, design};
    use super::*;
    use ndarray::{Array1, Array2, s};
    use p2a_validation::{Check, Tol};

    #[test]
    fn test_validate_qr_dqrdc2_aliasing() {
        let (case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let n = c.n();
        let x = design(&c, &["1", "x1", "x3", "x2", "x4"]); // x3 = x1 - 2 x2 in the CSV
        let y = c.numeric_array("yc");
        let qr = Qr::decompose(x.view(), LM_QR_TOL).unwrap();
        let k = qr.rank();
        let pivot: Vec<f64> = qr.pivot().iter().map(|&j| (j + 1) as f64).collect();
        let fit = least_squares(x.view(), y.view()).unwrap();
        let rss = fit.residuals.mapv(|r| r * r).sum();
        let s2 = rss / fit.df_residual as f64;
        let se: Vec<f64> = fit
            .unscaled_cov
            .diag()
            .iter()
            .map(|v| (v * s2).sqrt())
            .collect();
        // |R_ii| is unique for a given pivot; its sign is a convention (not pinned).
        let r_abs: Vec<f64> = qr.r_diag().iter().take(k).map(|v| v.abs()).collect();
        Check::new(&case)
            .scalar("qr_rank", k as f64, Tol::exact())
            .vector("qr_pivot", &pivot, Tol::exact())
            .vector("lmfit_coef", &fit.coefficients.to_vec(), Tol::coef(n)) // NaN == R NA
            .scalar("lmfit_rank", fit.rank as f64, Tol::exact())
            .scalar("lmfit_df", fit.df_residual as f64, Tol::exact())
            .scalar("lmfit_resid_ss", rss, Tol::stat(n))
            .vector("lm_se", &se, Tol::se(n)) // NaN for x2, as vcov(lm)
            .finish();
        let r_case = case.vector("qr_r_diag_rank");
        assert_eq!(r_abs.len(), r_case.len());
        for (got, want) in r_abs.iter().zip(&r_case) {
            assert!(
                Tol::coef(n).allows(*got, want.abs()),
                "|R_ii| {got} vs R {want}"
            );
        }
        assert_eq!(qr.aliased(), vec![3]);
        assert_eq!(fit.aliased, vec![3]);
        let names: Vec<String> = ["(Intercept)", "x1", "x3", "x2", "x4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            fit.aliasing_warning(&names).as_deref(),
            Some("Coefficients not defined because of singularities: x2")
        );
        // R[5,5] of the aliased column is rounding noise (R: -4.53e-15)
        let x2_norm = x.column(3).dot(&x.column(3)).sqrt();
        assert!(qr.r_diag()[4].abs() < LM_QR_TOL * x2_norm);
        // conditioning indicator from R's own diag(R)
        let rd = case.vector("qr_r_diag_rank");
        let ratio = rd.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
            / rd.iter().fold(f64::INFINITY, |a, v| a.min(v.abs()));
        assert!(Tol::coef(n).allows(qr.r_diag_ratio(), ratio));
        // fitted + residuals = y (lm.fit: fitted = y - residuals)
        for i in 0..n {
            assert!((fit.fitted[i] + fit.residuals[i] - y[i]).abs() <= 1e-12 * y[i].abs().max(1.0));
        }
    }

    #[test]
    fn test_validate_qr_edge_shapes() {
        let (case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let n = c.n();
        // cbind(1, x1, 2 * x1, x2, 0, x4): the rescaled duplicate and the zero column move
        // to the end (R pivot 1 2 4 6 3 5, rank 4)
        let mut x2 = design(&c, &["1", "x1", "x1", "x2", "x4", "x4"]);
        x2.column_mut(2).mapv_inplace(|v| 2.0 * v);
        x2.column_mut(4).fill(0.0);
        let q2 = Qr::decompose(x2.view(), LM_QR_TOL).unwrap();
        let p2: Vec<f64> = q2.pivot().iter().map(|&j| (j + 1) as f64).collect();
        // n < p: the first three rows of [1, x1, x3, x2, x4]
        let x = design(&c, &["1", "x1", "x3", "x2", "x4"]);
        let q3 = Qr::decompose(x.slice(s![0..3, ..]), LM_QR_TOL).unwrap();
        let p3: Vec<f64> = q3.pivot().iter().map(|&j| (j + 1) as f64).collect();
        Check::new(&case)
            .scalar("qr2_rank", q2.rank() as f64, Tol::exact())
            .vector("qr2_pivot", &p2, Tol::exact())
            .scalar("qr3_rank", q3.rank() as f64, Tol::exact())
            .vector("qr3_pivot", &p3, Tol::exact())
            .finish();
        assert_eq!(q2.aliased(), vec![2, 4]);
        assert_eq!(q3.qraux().len(), 5);
        // p = 0: rank 0, residuals = y, df = n
        let y = c.numeric_array("yc");
        let empty = Array2::<f64>::zeros((n, 0));
        let f0 = least_squares(empty.view(), y.view()).unwrap();
        assert_eq!((f0.rank, f0.df_residual, f0.coefficients.len()), (0, n, 0));
        assert_eq!(f0.residuals, y);
    }

    #[test]
    fn test_validate_lm_wfit_zero_weights() {
        let (case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let n = c.n();
        let x = design(&c, &["1", "x1", "x2"]);
        let (y, w) = (c.numeric_array("yc"), c.numeric_array("w0")); // w0[0..5] == 0
        let fit = weighted_least_squares(x.view(), y.view(), w.view()).unwrap();
        let s2 = (0..n)
            .filter(|&i| w[i] > 0.0)
            .map(|i| w[i] * fit.residuals[i].powi(2))
            .sum::<f64>()
            / fit.df_residual as f64;
        let se: Vec<f64> = fit
            .unscaled_cov
            .diag()
            .iter()
            .map(|v| (v * s2).sqrt())
            .collect();
        Check::new(&case)
            .vector("lmwfit_coef", &fit.coefficients.to_vec(), Tol::coef(n))
            .scalar("lmwfit_df", fit.df_residual as f64, Tol::exact()) // 112 = 115 positive weights - 3
            .vector(
                "lmwfit_resid_head",
                &fit.residuals.to_vec()[..8],
                Tol::coef(n),
            ) // rows 1-5: y0 - x0 b
            .vector(
                "lmwfit_fitted_head",
                &fit.fitted.to_vec()[..8],
                Tol::coef(n),
            )
            .vector("lmw_se", &se, Tol::se(n))
            .finish();
        assert_eq!((fit.residuals.len(), fit.fitted.len()), (n, n));
    }

    /// QR identities on the three designs of the case (properties, not R-pinned).
    #[test]
    fn test_qr_factorization_properties() {
        let (_case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let n = c.n();
        let x = design(&c, &["1", "x1", "x3", "x2", "x4"]);
        let mut x2 = design(&c, &["1", "x1", "x1", "x2", "x4", "x4"]);
        x2.column_mut(2).mapv_inplace(|v| 2.0 * v);
        x2.column_mut(4).fill(0.0);
        let x3 = x.slice(s![0..3, ..]).to_owned();
        let yc = c.numeric_array("yc");
        let x1 = c.numeric_array("x1");
        for xm in [&x, &x2, &x3] {
            let rows = xm.nrows();
            let qr = Qr::decompose(xm.view(), LM_QR_TOL).unwrap();
            let k = qr.rank();
            let big = xm.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            // (a) Q [R[0..=j, j]; 0] = X[:, pivot[j]]
            for j in 0..k {
                let mut v = Array1::<f64>::zeros(rows);
                for i in 0..=j {
                    v[i] = qr.r(i, j);
                }
                let col = qr.q_times(v.view()).unwrap();
                let want = xm.column(qr.pivot()[j]);
                for i in 0..rows {
                    assert!(
                        (col[i] - want[i]).abs() <= 1e-12 * big,
                        "column {j}, row {i}: {} vs {}",
                        col[i],
                        want[i]
                    );
                }
            }
            // (b) an aliased column is (numerically) in the span of the identified ones
            let ident: Vec<usize> = qr.pivot()[..k].to_vec();
            let xi = xm.select(ndarray::Axis(1), &ident);
            for &j in &qr.aliased() {
                let xj = xm.column(j);
                let norm = xj.dot(&xj).sqrt();
                if rows <= k {
                    continue; // exact interpolation: nothing to test
                }
                let f = least_squares(xi.view(), xj).unwrap();
                let r = f.residuals.dot(&f.residuals).sqrt();
                assert!(
                    r <= LM_QR_TOL * norm,
                    "aliased column {j}: residual {r} vs {norm}"
                );
            }
            // (c) Q is orthogonal: qty(q_times(v)) == v and the norm is kept
            for v in [&yc, &x1] {
                let v = v.slice(s![..rows]).to_owned();
                let qv = qr.q_times(v.view()).unwrap();
                let back = qr.qty(qv.view()).unwrap();
                let nv = v.dot(&v).sqrt();
                assert!((qv.dot(&qv).sqrt() - nv).abs() <= 1e-12 * nv);
                for i in 0..rows {
                    assert!((back[i] - v[i]).abs() <= 1e-12 * nv);
                }
            }
        }
        assert_eq!(n, 120);
    }

    /// Shared by the degenerate-input tests of this module.
    fn spec_err<T: std::fmt::Debug>(r: EconResult<T>, needle: &str) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains(needle), "{message}")
            }
            other => panic!("expected InvalidSpecification containing {needle:?}, got {other:?}"),
        }
    }

    #[test]
    fn test_least_squares_rejects_invalid_input() {
        let x = ndarray::array![[1.0, 0.3], [1.0, 1.2], [1.0, 2.1], [1.0, 2.9]];
        let y = ndarray::array![1.1, 1.9, 3.2, 3.8];
        let mut xn = x.clone();
        xn[[2, 1]] = f64::NAN;
        spec_err(least_squares(xn.view(), y.view()), "x");
        let mut yi = y.clone();
        yi[1] = f64::INFINITY;
        spec_err(least_squares(x.view(), yi.view()), "y");
        spec_err(least_squares(x.view(), y.slice(s![..3])), "rows");
        spec_err(least_squares_tol(x.view(), y.view(), 0.0), "tolerance");
        spec_err(least_squares_tol(x.view(), y.view(), f64::NAN), "tolerance");
        spec_err(
            weighted_least_squares(
                x.view(),
                y.view(),
                ndarray::array![1.0, -0.5, 1.0, 1.0].view(),
            ),
            "weights",
        );
        spec_err(
            weighted_least_squares(
                x.view(),
                y.view(),
                ndarray::array![1.0, f64::NAN, 1.0, 1.0].view(),
            ),
            "weights",
        );
        // a non-finite x in a zero-weight row is refused too (lm.wfit)
        let mut xz = x.clone();
        xz[[1, 1]] = f64::INFINITY;
        spec_err(
            weighted_least_squares(
                xz.view(),
                y.view(),
                ndarray::array![1.0, 0.0, 1.0, 1.0].view(),
            ),
            "x contains a non-finite value (row 1, column 1)",
        );
        assert!(matches!(
            weighted_least_squares(x.view(), y.view(), Array1::<f64>::zeros(4).view()),
            Err(EconError::InsufficientData { .. })
        ));
        // Qr's solves refuse a y of the wrong length instead of panicking
        let qr = Qr::decompose(x.view(), LM_QR_TOL).unwrap();
        spec_err(qr.qty(y.slice(s![..3])), "rows");
        spec_err(qr.coefficients(y.slice(s![..3])), "rows");
        spec_err(qr.q_times(y.slice(s![..3])), "rows");
        // a finite weight whose square root overflows the scaled row: the caller's row and
        // the weight are named (not a row of the compacted positive-weight matrix)
        let mut xb = x.clone();
        xb[[2, 1]] = 1e200;
        spec_err(
            weighted_least_squares(
                xb.view(),
                y.view(),
                ndarray::array![0.0, 1.0, 1e300, 1.0].view(),
            ),
            "weights: row 2 (w = 1e300) overflows the weighted design",
        );
        let mut yb = y.clone();
        yb[3] = 1e200;
        spec_err(
            weighted_least_squares(
                x.view(),
                yb.view(),
                ndarray::array![0.0, 1.0, 1.0, 1e300].view(),
            ),
            "weights: row 3",
        );
        // the p x p covariance of a wide x goes through the allocation ceiling
        let wide = Array2::from_shape_fn((2, 64), |(i, j)| ((i + 1) * (j + 3) % 7) as f64 + 0.1);
        let qw = Qr::decompose(wide.view(), LM_QR_TOL).unwrap();
        assert!(qw.unscaled_covariance().is_ok());
        assert!(matches!(
            crate::resource::with_alloc_ceiling(Some(4096), || qw.unscaled_covariance()),
            Err(EconError::ResourceLimit(_))
        ));
        assert!(matches!(
            least_squares(
                Array2::<f64>::zeros((0, 2)).view(),
                Array1::<f64>::zeros(0).view()
            ),
            Err(EconError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_sym_inverse_reports_rank() {
        let (case, ds) = crate::test_support::load_case("linalg_qr_glm");
        let c = columns(&ds);
        let n = c.n();
        let xt = design(&c, &["1", "x1", "ysep", "1-ysep"]); // intercept + both dummies
        let y = c.numeric_array("yc");
        let m = crate::linalg::matrix_ops::xtx(&xt.view());
        let inv = sym_inverse(m.view(), SingularPolicy::Generalized, "X'X").unwrap();
        assert_eq!(
            (inv.rank, inv.dim, inv.generalized, inv.is_full_rank()),
            (3, 4, true, false)
        );
        assert_eq!(
            inv.warning("X'X").as_deref(),
            Some("X'X is singular (rank 3 of 4); a generalized inverse was used")
        );
        let b = inv.inverse.dot(&xt.t().dot(&y));
        let rss = (&y - &xt.dot(&b)).mapv(|r| r * r).sum();
        Check::new(&case)
            .scalar("trap_rank", inv.rank as f64, Tol::exact())
            .scalar("trap_x1_coef", b[1], Tol::coef(n)) // estimable: identical for every generalized inverse
            .scalar("trap_rss", rss, Tol::stat(n))
            .finish();
        // reflexive symmetric generalized inverse (R emulation: 3.7e-15 and 3.4e-15)
        let g = &inv.inverse;
        let mx = |a: &Array2<f64>| a.iter().fold(0.0_f64, |s, v| s.max(v.abs()));
        assert!(mx(&(m.dot(g).dot(&m) - &m)) <= 1e-10 * mx(&m));
        assert!(mx(&(g.dot(&m).dot(g) - g)) <= 1e-10 * mx(g));
        assert!(mx(&(g - &g.t())) <= 1e-12 * mx(g));
        match sym_inverse(m.view(), SingularPolicy::Error, "X'X") {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert_eq!(context, "X'X: rank 3 of 4")
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
        // safe_inverse now flags the dummy trap. Before: None whenever faer's Cholesky
        // passed the zero pivot (data-dependent; REG-4 observed both outcomes). The
        // deterministic fails-before test is
        // matrix_ops::tests::test_safe_inverse_warns_on_near_singular_pd.
        let (_, warn) = crate::linalg::matrix_ops::safe_inverse(&m.view()).unwrap();
        assert!(warn.is_some_and(|c| c > crate::linalg::matrix_ops::CONDITION_THRESHOLD));
    }

    #[test]
    fn test_sym_inverse_rejects_nonsymmetric_and_nonfinite() {
        spec_err(
            sym_inverse(
                ndarray::array![[2.0, 1.0], [0.0, 3.0]].view(),
                SingularPolicy::Error,
                "A",
            ),
            "not symmetric",
        );
        spec_err(
            sym_inverse(
                ndarray::array![[1.0, f64::NAN], [f64::NAN, 1.0]].view(),
                SingularPolicy::Error,
                "A",
            ),
            "non-finite",
        );
        assert!(matches!(
            sym_inverse(
                Array2::<f64>::zeros((2, 3)).view(),
                SingularPolicy::Error,
                "A"
            ),
            Err(EconError::LinalgError(
                crate::linalg::LinalgError::NotSquare { .. }
            ))
        ));
        // asymmetry at 1e-13 relative is rounding, not asymmetry
        let m = ndarray::array![[4.0, 1.0 + 1e-13], [1.0, 3.0]];
        assert!(
            sym_inverse(m.view(), SingularPolicy::Error, "A")
                .unwrap()
                .is_full_rank()
        );
        // the zero matrix: rank 0; Error refuses, Generalized returns the zero matrix
        assert!(matches!(
            sym_inverse(
                Array2::<f64>::zeros((2, 2)).view(),
                SingularPolicy::Error,
                "Z"
            ),
            Err(EconError::SingularMatrix { .. })
        ));
        let z = sym_inverse(
            Array2::<f64>::zeros((2, 2)).view(),
            SingularPolicy::Generalized,
            "Z",
        )
        .unwrap();
        assert_eq!((z.rank, z.reciprocal_condition), (0, 0.0));
        assert!(z.inverse.iter().all(|v| *v == 0.0));
        let e = sym_inverse(
            Array2::<f64>::zeros((0, 0)).view(),
            SingularPolicy::Error,
            "E",
        )
        .unwrap();
        assert_eq!((e.rank, e.dim, e.generalized), (0, 0, false));
    }

    #[test]
    fn test_sym_inverse_negative_definite_and_indefinite() {
        // a negated Hessian passed un-negated (binary_choice.rs, count_models.rs do this):
        // the Cholesky factor of -H inverts it
        let h = ndarray::array![[-4.0, 1.0], [1.0, -3.0]];
        let s = sym_inverse(h.view(), SingularPolicy::Error, "H").unwrap();
        let r = ndarray::array![
            [-0.2727272727272727, -0.09090909090909091],
            [-0.09090909090909091, -0.36363636363636365]
        ];
        for (g, w) in s.inverse.iter().zip(r.iter()) {
            assert!((g - w).abs() <= 1e-15);
        }
        assert!((s.reciprocal_condition - 0.5519815245204085).abs() <= 1e-12); // R, equilibrated by |diag|
        // KKT matrix with a zero diagonal entry: full rank (scale 1 for the zero diagonal)
        let k = ndarray::array![[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [1.0, 1.0, 0.0]];
        let s = sym_inverse(k.view(), SingularPolicy::Error, "KKT").unwrap();
        assert_eq!((s.rank, s.generalized), (3, false));
        assert!((s.reciprocal_condition - 0.38196601125010504).abs() <= 1e-12);
        // indefinite with every diagonal entry negative: the Cholesky of -m fails too, and
        // the LU inverse is returned bit for bit
        let m = ndarray::array![[-1.0, 2.1, 0.3], [2.1, -1.2, 0.4], [0.3, 0.4, -0.9]];
        let s = sym_inverse(m.view(), SingularPolicy::Error, "M").unwrap();
        let lu = crate::linalg::matrix_ops::partial_piv_lu_inverse(&to_faer(&m.view()));
        assert_eq!(s.inverse, from_faer(&lu));
        // a badly scaled (graded) indefinite matrix: the kappa_1 allowance still accepts
        // the LU inverse (it is within a factor dim of the exact 2-norm condition)
        let d = 6;
        let g: Vec<f64> = (0..d).map(|i| 10f64.powi(3 * i as i32 - 6)).collect();
        let base = Array2::from_shape_fn((d, d), |(i, j)| {
            let o = 0.3 * ((i * 5 + j * 5) % 7) as f64 / 7.0;
            o + if i == j {
                if i % 2 == 0 { 2.0 } else { -2.0 }
            } else {
                0.0
            }
        });
        let m = Array2::from_shape_fn((d, d), |(i, j)| g[i] * base[[i, j]] * g[j]);
        let s = sym_inverse(m.view(), SingularPolicy::Error, "M").unwrap();
        assert!(s.is_full_rank());
        let lu = crate::linalg::matrix_ops::partial_piv_lu_inverse(&to_faer(&m.view()));
        assert_eq!(s.inverse, from_faer(&lu));
        // the candidate-independent cap binds: an allowance for cond 1 refuses this LU
        let a = to_faer(&m.view());
        assert!(crate::linalg::matrix_ops::lu_inverse_bounded(&a, f64::INFINITY).is_some());
        assert!(crate::linalg::matrix_ops::lu_inverse_bounded(&a, 1.0).is_none());
    }

    #[test]
    fn test_finite_asymmetry_layouts() {
        let m = ndarray::array![[4.0, 1.0, -2.5], [1.5, 3.0, 0.2], [-2.5, 0.2, 7.0]];
        let want = Some((0.5, 7.0));
        assert_eq!(finite_asymmetry(&m.view()), want);
        assert_eq!(finite_asymmetry(&m.t()), want);
        assert_eq!(finite_asymmetry(&m.slice(s![..;1, ..;1])), want);
        for v in [f64::NAN, f64::INFINITY] {
            for (i, j) in [(0, 0), (0, 2), (2, 1)] {
                let mut b = m.clone();
                b[[i, j]] = v;
                assert_eq!(finite_asymmetry(&b.view()), None, "({i}, {j}) = {v}");
                assert_eq!(
                    finite_asymmetry(&b.t()),
                    None,
                    "({i}, {j}) = {v}, transposed"
                );
            }
        }
        // remainder of the four-lane pass (5 x 5 = 25 entries) and d = 0, 1
        let mut f = Array2::<f64>::eye(5);
        f[[4, 4]] = f64::NAN;
        assert_eq!(finite_asymmetry(&f.view()), None);
        assert_eq!(
            finite_asymmetry(&Array2::<f64>::zeros((0, 0)).view()),
            Some((0.0, 0.0))
        );
        assert_eq!(
            finite_asymmetry(&ndarray::array![[-3.0]].view()),
            Some((0.0, 3.0))
        );
    }
}
