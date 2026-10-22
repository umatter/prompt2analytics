//! Resource guards for caller-sized work: fallible allocation, checked size
//! arithmetic, and cooperative cancellation (deadline or cancel flag) for long
//! loops.
//!
//! # Allocation guards
//!
//! Buffers whose size comes from a caller (a replication count, `n * n`, a
//! grid size) go through [`try_vec`], [`try_with_capacity`], [`try_zeros1`] or
//! [`try_zeros2`]. An impossible request (the byte count overflows or exceeds
//! `isize::MAX`) or one above the process-wide per-allocation ceiling
//! ([`set_alloc_ceiling_bytes`]) becomes [`EconError::ResourceLimit`] instead
//! of a process abort. The ceiling is off (unlimited) by default; servers
//! install one at startup.
//!
//! # Cooperative cancellation
//!
//! A [`Budget`] (an optional deadline plus a [`CancelToken`]) can be installed
//! for the current thread with [`with_budget`]. Long loops call
//! [`checkpoint`] once per outer iteration; it returns
//! [`EconError::Cancelled`] once the budget's token is cancelled or its
//! deadline has passed, and `Ok(())` when no budget is installed. Inside rayon
//! closures, capture [`current_budget`] before the parallel section and call
//! [`Budget::check`] on the captured value.
//!
//! # Example
//!
//! ```
//! use p2a_core::resource::{Budget, CancelToken, checkpoint, try_zeros2, with_budget};
//!
//! let m = try_zeros2(3, 4, "design matrix").unwrap();
//! assert_eq!(m.dim(), (3, 4));
//! assert!(try_zeros2(1 << 40, 1 << 40, "dense matrix").is_err());
//!
//! let token = CancelToken::new();
//! token.cancel();
//! let stopped = with_budget(Budget::new(None, token), || {
//!     for _ in 0..1_000_000 {
//!         checkpoint()?;
//!     }
//!     Ok::<(), p2a_core::EconError>(())
//! });
//! assert!(stopped.is_err());
//! ```

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2};

use crate::errors::{EconError, EconResult};
use crate::memory::format_bytes;

// ═══════════════════════════════════════════════════════════════════════════
// Allocation ceiling
// ═══════════════════════════════════════════════════════════════════════════

/// Process-wide per-allocation ceiling in bytes; `usize::MAX` means none.
static ALLOC_CEILING: AtomicUsize = AtomicUsize::new(usize::MAX);

thread_local! {
    /// Thread-local override of the ceiling (`Some(limit)`), used by tests.
    static ALLOC_CEILING_OVERRIDE: Cell<Option<Option<usize>>> = const { Cell::new(None) };
}

/// Set the process-wide per-allocation ceiling in bytes (`None` = unlimited).
///
/// Servers call this once at startup from their resource profile. The
/// default is unlimited, so library and CLI use is unconstrained.
pub fn set_alloc_ceiling_bytes(limit: Option<usize>) {
    ALLOC_CEILING.store(limit.unwrap_or(usize::MAX), Ordering::Relaxed);
}

/// The effective per-allocation ceiling for the current thread.
///
/// A thread-local override installed by [`with_alloc_ceiling`] wins;
/// otherwise this is the process-wide value from [`set_alloc_ceiling_bytes`].
pub fn alloc_ceiling_bytes() -> Option<usize> {
    if let Some(limit) = ALLOC_CEILING_OVERRIDE.with(Cell::get) {
        return limit;
    }
    match ALLOC_CEILING.load(Ordering::Relaxed) {
        usize::MAX => None,
        limit => Some(limit),
    }
}

/// Run `f` with a thread-local per-allocation ceiling, restoring the previous
/// setting afterwards (also when `f` panics).
///
/// Meant for tests, which run in parallel and must not change the
/// process-wide value.
pub fn with_alloc_ceiling<R>(limit: Option<usize>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Option<usize>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ALLOC_CEILING_OVERRIDE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(ALLOC_CEILING_OVERRIDE.with(|c| c.replace(Some(limit))));
    f()
}

// ═══════════════════════════════════════════════════════════════════════════
// Checked sizes and fallible allocation
// ═══════════════════════════════════════════════════════════════════════════

/// The [`EconError::ResourceLimit`] for a size computation on `what` that
/// overflows.
pub(crate) fn overflow(what: &str) -> EconError {
    EconError::ResourceLimit(format!("{what} is too large: the requested size overflows"))
}

/// The product of `dims` (`1` for an empty slice), or
/// [`EconError::ResourceLimit`] when it overflows `usize`.
pub fn checked_len(dims: &[usize], what: &str) -> EconResult<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| overflow(what))
}

/// Check that a buffer of `len` values of `T` can be allocated: the byte count
/// must not overflow, must not exceed `isize::MAX`, and must not exceed the
/// effective per-allocation ceiling.
pub fn check_alloc<T>(len: usize, what: &str) -> EconResult<()> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| overflow(what))?;
    if bytes > isize::MAX as usize {
        return Err(overflow(what));
    }
    if let Some(limit) = alloc_ceiling_bytes()
        && bytes > limit
    {
        return Err(EconError::ResourceLimit(format!(
            "{what} would need {}, above this server's per-allocation limit of {}; \
             use a smaller input",
            format_bytes(bytes),
            format_bytes(limit)
        )));
    }
    Ok(())
}

/// The [`EconError::ResourceLimit`] for an allocation of `len` values of
/// `size` bytes for `what` that passed the checks but could not be made.
pub(crate) fn reserve_failed(what: &str, len: usize, size: usize) -> EconError {
    EconError::ResourceLimit(format!(
        "{what} would need {}, which could not be allocated; use a smaller input",
        format_bytes(len.saturating_mul(size))
    ))
}

/// A vector of `len` copies of `fill`, allocated fallibly.
pub fn try_vec<T: Clone>(len: usize, fill: T, what: &str) -> EconResult<Vec<T>> {
    let mut v = try_with_capacity::<T>(len, what)?;
    v.resize(len, fill);
    Ok(v)
}

/// An empty vector with room for at least `cap` values, allocated fallibly.
pub fn try_with_capacity<T>(cap: usize, what: &str) -> EconResult<Vec<T>> {
    check_alloc::<T>(cap, what)?;
    let mut v = Vec::new();
    v.try_reserve_exact(cap)
        .map_err(|_| reserve_failed(what, cap, std::mem::size_of::<T>()))?;
    Ok(v)
}

/// A zero vector of length `len`, allocated fallibly.
pub fn try_zeros1(len: usize, what: &str) -> EconResult<Array1<f64>> {
    Ok(Array1::from_vec(try_vec(len, 0.0, what)?))
}

/// A `rows x cols` zero matrix, allocated fallibly.
pub fn try_zeros2(rows: usize, cols: usize, what: &str) -> EconResult<Array2<f64>> {
    let len = checked_len(&[rows, cols], what)?;
    let data = try_vec(len, 0.0, what)?;
    Array2::from_shape_vec((rows, cols), data).map_err(|_| overflow(what))
}

// ═══════════════════════════════════════════════════════════════════════════
// Cooperative cancellation
// ═══════════════════════════════════════════════════════════════════════════

/// A shared cancellation flag. Clones observe the same flag.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A new token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel the token; every clone sees it.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether [`cancel`](Self::cancel) has been called on any clone.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A computation budget: an optional wall-clock deadline plus a cancel token.
#[derive(Clone, Debug)]
pub struct Budget {
    deadline: Option<Instant>,
    token: CancelToken,
}

impl Budget {
    /// A budget that expires at `deadline` (if any) or when `token` is cancelled.
    pub fn new(deadline: Option<Instant>, token: CancelToken) -> Self {
        Self { deadline, token }
    }

    /// A budget that never expires (its token is never cancelled by anyone else).
    pub fn unlimited() -> Self {
        Self::new(None, CancelToken::new())
    }

    /// `Err(EconError::Cancelled)` once the token is cancelled or the deadline
    /// has passed; `Ok(())` otherwise.
    pub fn check(&self) -> EconResult<()> {
        if self.token.is_cancelled() {
            return Err(EconError::Cancelled(
                "the computation was cancelled".to_string(),
            ));
        }
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return Err(EconError::Cancelled(
                "the time limit for this computation was reached".to_string(),
            ));
        }
        Ok(())
    }

    /// The deadline, if the budget has one.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
}

thread_local! {
    static CURRENT_BUDGET: RefCell<Option<Budget>> = const { RefCell::new(None) };
}

/// Run `f` with `budget` installed for the current thread, restoring the
/// previously installed budget afterwards (also when `f` panics).
pub fn with_budget<R>(budget: Budget, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Budget>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            CURRENT_BUDGET.with(|b| *b.borrow_mut() = prev);
        }
    }
    let prev = CURRENT_BUDGET.with(|b| b.borrow_mut().replace(budget));
    let _restore = Restore(prev);
    f()
}

/// The budget installed for the current thread, if any. Capture it before a
/// rayon parallel section and call [`Budget::check`] inside the closures.
pub fn current_budget() -> Option<Budget> {
    CURRENT_BUDGET.with(|b| b.borrow().clone())
}

/// Check the current thread's budget: `Err(EconError::Cancelled)` when it is
/// cancelled or past its deadline, `Ok(())` when it is not or when no budget
/// is installed.
pub fn checkpoint() -> EconResult<()> {
    CURRENT_BUDGET.with(|b| match b.borrow().as_ref() {
        Some(budget) => budget.check(),
        None => Ok(()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_len_detects_overflow() {
        assert_eq!(checked_len(&[3, 4, 5], "t").unwrap(), 60);
        assert_eq!(checked_len(&[], "t").unwrap(), 1);
        let e = checked_len(&[usize::MAX, 2], "lag matrix").unwrap_err();
        assert!(matches!(&e, EconError::ResourceLimit(m) if m.contains("lag matrix")));
    }

    #[test]
    fn impossible_allocation_is_an_error_not_an_abort() {
        // 2^60 f64 = 2^63 bytes > isize::MAX: must return Err without touching the allocator
        assert!(matches!(
            try_vec::<f64>(1usize << 60, 0.0, "buf"),
            Err(EconError::ResourceLimit(_))
        ));
        assert!(matches!(
            try_zeros2(1usize << 31, 1usize << 31, "dense"),
            Err(EconError::ResourceLimit(_))
        ));
        assert!(matches!(
            check_alloc::<f64>(usize::MAX / 4, "x"),
            Err(EconError::ResourceLimit(_))
        ));
    }

    #[test]
    fn ceiling_applies_per_thread_override() {
        with_alloc_ceiling(Some(1024), || {
            let e = try_vec::<f64>(1000, 0.0, "grid").unwrap_err(); // 8000 bytes > 1024
            let m = e.to_string();
            assert!(m.contains("grid") && m.contains("limit"), "{m}");
            assert!(try_vec::<f64>(100, 0.0, "grid").is_ok()); // 800 bytes
        });
        assert!(try_vec::<f64>(1000, 0.0, "grid").is_ok()); // override restored
    }

    #[test]
    fn global_ceiling_round_trips() {
        // harmless value: other tests in this binary run in parallel
        set_alloc_ceiling_bytes(Some(usize::MAX - 1));
        assert_eq!(alloc_ceiling_bytes(), Some(usize::MAX - 1));
        set_alloc_ceiling_bytes(None);
        assert_eq!(alloc_ceiling_bytes(), None);
    }

    #[test]
    fn zeros_have_requested_shape() {
        assert_eq!(try_zeros2(3, 4, "m").unwrap().dim(), (3, 4));
        assert_eq!(try_zeros1(5, "v").unwrap().len(), 5);
        assert!(try_with_capacity::<u8>(16, "c").unwrap().capacity() >= 16);
    }

    #[test]
    fn budget_checks_token_and_deadline() {
        assert!(Budget::unlimited().check().is_ok());
        let t = CancelToken::new();
        let b = Budget::new(None, t.clone());
        assert!(b.check().is_ok());
        t.cancel();
        assert!(matches!(b.check(), Err(EconError::Cancelled(m)) if m.contains("cancelled")));
        let past = Budget::new(Some(Instant::now()), CancelToken::new());
        assert!(matches!(past.check(), Err(EconError::Cancelled(m)) if m.contains("time limit")));
    }

    #[test]
    fn checkpoint_sees_installed_budget_and_restores() {
        assert!(checkpoint().is_ok());
        let t = CancelToken::new();
        t.cancel();
        let inner = with_budget(Budget::new(None, t), || {
            let nested_ok = with_budget(Budget::unlimited(), checkpoint).is_ok();
            (nested_ok, checkpoint().is_err())
        });
        assert_eq!(inner, (true, true));
        assert!(checkpoint().is_ok());
        let _ = std::panic::catch_unwind(|| with_budget(Budget::unlimited(), || panic!("boom")));
        assert!(current_budget().is_none());
    }

    #[test]
    fn captured_budget_works_on_other_threads() {
        // the rayon pattern
        let t = CancelToken::new();
        let b = with_budget(Budget::new(None, t.clone()), || current_budget().unwrap());
        t.cancel();
        assert!(
            std::thread::spawn(move || b.check())
                .join()
                .unwrap()
                .is_err()
        );
    }
}
