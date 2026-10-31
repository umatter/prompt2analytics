//! Utility functions for panel data estimation.
//!
//! Contains helper functions for entity ID extraction, demeaning,
//! and other operations common to panel data estimators.

use std::collections::BTreeSet;
use std::ops::Range;

use ndarray::{Array1, Array2};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, Columns, Factor, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource;

/// Compute entity-level means for demeaning.
pub fn compute_entity_means(
    data: &Array1<f64>,
    entity_ids: &[usize],
    n_groups: usize,
) -> Array1<f64> {
    let n = data.len();
    let mut group_sums = vec![0.0; n_groups];
    let mut group_counts = vec![0usize; n_groups];

    for (i, &val) in data.iter().enumerate() {
        let g = entity_ids[i];
        group_sums[g] += val;
        group_counts[g] += 1;
    }

    let group_means: Vec<f64> = group_sums
        .iter()
        .zip(group_counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect();

    // Create array with entity means for each observation
    let mut means = Array1::zeros(n);
    for i in 0..n {
        means[i] = group_means[entity_ids[i]];
    }
    means
}

/// Demean a vector by entity (for Fixed Effects).
pub fn demean_by_entity(data: &Array1<f64>, entity_ids: &[usize], n_groups: usize) -> Array1<f64> {
    let means = compute_entity_means(data, entity_ids, n_groups);
    data - &means
}

/// Demean a matrix by entity (for Fixed Effects).
pub fn demean_matrix_by_entity(
    x: &Array2<f64>,
    entity_ids: &[usize],
    n_groups: usize,
) -> Array2<f64> {
    let (n, k) = x.dim();
    let mut x_demeaned = Array2::zeros((n, k));

    for j in 0..k {
        let col = x.column(j).to_owned();
        let col_demeaned = demean_by_entity(&col, entity_ids, n_groups);
        x_demeaned.column_mut(j).assign(&col_demeaned);
    }

    x_demeaned
}

/// Entity codes of the kept rows (0-based, in `factor.levels` order: R factor() order,
/// numeric ids numerically, text ids in C order) and the number of entities present.
pub(crate) fn entity_codes(factor: &Factor) -> (Vec<usize>, usize) {
    let codes = factor.codes.iter().map(|&c| c as usize).collect();
    (codes, factor.n_levels())
}

/// Dense period index (0 = earliest) of each time value, and the sorted distinct values.
/// O(n log n): a sorted, de-duplicated copy and a binary search per value.
pub(crate) fn time_codes(times: &[i64]) -> (Vec<usize>, Vec<i64>) {
    let mut distinct = times.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let codes = times
        .iter()
        .map(|t| distinct.binary_search(t).unwrap_or_default())
        .collect();
    (codes, distinct)
}

/// Lag specification of a difference-GMM model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GmmLagSpec {
    /// Lags of the dependent variable among the regressors.
    pub lags: usize,
    /// Shallowest lag of y used as instrument.
    pub min_lag: usize,
    /// Deepest lag of y used as instrument; `None`: every available lag.
    pub max_lag: Option<usize>,
    /// One instrument column per lag depth instead of one per (period, lag).
    pub collapse: bool,
}

/// The first-differenced equations of a dynamic panel in plm::pgmm's layout.
pub(crate) struct DiffEquations {
    /// Δy_t, one per equation.
    pub dy: Vec<f64>,
    /// `[Δy_{t-1} .. Δy_{t-lags}, Δx_t]`, n_eq x k.
    pub x: Array2<f64>,
    /// `[GMM columns, Δx_t]`, n_eq x L.
    pub z: Array2<f64>,
    /// Number of GMM columns (L - k_x).
    pub n_gmm_instruments: usize,
    /// t of each equation.
    pub time: Vec<i64>,
    /// Equation rows of each entity with at least one equation, in level order.
    pub blocks: Vec<Range<usize>>,
    /// Factor label of each block.
    pub block_labels: Vec<String>,
    /// Labels of the entities without an equation, in level order.
    pub entities_without_equations: Vec<String>,
    /// t_max - t_min + 1.
    pub span: usize,
}

/// Greatest common divisor of two positive numbers.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// The first-differenced equations of `y` on its `spec.lags` lags and `x`, in the
/// layout of `plm::pgmm(transformation = "d")`.
///
/// `cols` must hold `entity` (factor), `time` (time index), `y` and every `x`
/// (numeric), complete rows. Lags are steps of 1 in `time` (plm's time-aware `lag()`,
/// Stata without `delta()`): the equation of (entity, t) exists only where the entity
/// has y at t, t-1, ..., t-lags-1 and t - t_min >= max(lags + 1, min_lag). Its GMM
/// instruments are the entity's observed levels y_s, s in [t - max_lag, t - min_lag],
/// one column per (t, s), or per lag depth t - s with `collapse`; a column that no
/// equation observes is not created. The Δx columns follow as standard instruments.
/// Nothing is zero-filled except the instrument cells an equation does not have. A
/// repeated (entity, time) is an error.
pub(crate) fn difference_equations(
    cols: &Columns,
    y: &str,
    x: &[&str],
    entity: &str,
    time: &str,
    spec: &GmmLagSpec,
) -> EconResult<DiffEquations> {
    let factor = cols.factor(entity);
    let (codes, n_levels) = entity_codes(factor);
    let times = cols.time_index(time);
    let yv = cols.numeric(y);
    let xv: Vec<&[f64]> = x.iter().map(|name| cols.numeric(name)).collect();
    let n = codes.len();
    let k_x = x.len();

    // Rows grouped by entity in level order (a stable counting sort), then by time.
    let mut start = resource::try_vec(n_levels.saturating_add(1), 0usize, "GMM entity offsets")?;
    for &c in &codes {
        start[c + 1] += 1;
    }
    for e in 0..n_levels {
        start[e + 1] += start[e];
    }
    let mut next = resource::try_vec(n_levels, 0usize, "GMM entity offsets")?;
    next.copy_from_slice(&start[..n_levels]);
    let mut order = resource::try_vec(n, 0usize, "GMM row order")?;
    for (row, &c) in codes.iter().enumerate() {
        order[next[c]] = row;
        next[c] += 1;
    }
    drop(next);
    for e in 0..n_levels {
        resource::checkpoint()?;
        let group = &mut order[start[e]..start[e + 1]];
        group.sort_by_key(|&r| times[r]);
        let mut a = 0;
        while a < group.len() {
            let t = times[group[a]];
            let b = a + group[a..].partition_point(|&r| times[r] == t);
            if b - a > 1 {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Entity '{}' has {} rows for time {t}; difference GMM needs one row \
                         per (entity, time).",
                        factor.levels[e],
                        b - a
                    ),
                });
            }
            a = b;
        }
    }
    let mut st: Vec<i64> = resource::try_with_capacity(n, "GMM sorted times")?;
    st.extend(order.iter().map(|&r| times[r]));

    // The time span and the lag checks.
    let (Some(&t_min), Some(&t_max)) = (times.iter().min(), times.iter().max()) else {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "GMM estimation: no complete rows".to_string(),
        });
    };
    let too_wide = || EconError::InvalidSpecification {
        message: format!(
            "Time column '{time}' runs from {t_min} to {t_max}, more periods than can be \
             indexed; recode it as consecutive period numbers."
        ),
    };
    let span_i64 = t_max
        .checked_sub(t_min)
        .and_then(|d| d.checked_add(1))
        .ok_or_else(too_wide)?;
    let span = usize::try_from(span_i64).map_err(|_| too_wide())?;
    let lags = spec.lags;
    // The first equation is at period lags + 2 (y at t, ..., t - lags - 1), as in pgmm.
    if lags >= span.saturating_sub(1) {
        return Err(EconError::InsufficientData {
            required: lags.saturating_add(2),
            provided: span,
            context: format!(
                "GMM estimation requires at least {} time periods for {} lags",
                lags.saturating_add(2),
                lags
            ),
        });
    }
    if spec.min_lag == 0 {
        return Err(EconError::InvalidSpecification {
            message: "min_lag must be at least 1: lag 0 is the period's own level, which is \
                      correlated with the differenced error (the default is 2)."
                .to_string(),
        });
    }
    let max_lag_eff = spec.max_lag.unwrap_or(span - 1).min(span - 1);
    if spec.min_lag > max_lag_eff {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "max_lag ({max_lag_eff} after limiting to T - 1) must be at least min_lag \
                 ({}); T = {span} (the periods the time column spans)",
                spec.min_lag
            ),
        });
    }
    // lags < span - 1 and min_lag <= max_lag_eff < span, and span fits in i64.
    let k = lags + k_x;
    let lags_i = lags as i64;
    let min_lag = spec.min_lag as i64;
    let max_lag = max_lag_eff as i64;
    let tl1 = (lags_i + 1).max(min_lag);
    let collapse = spec.collapse;

    // An equation at sorted position j of an entity's times `ts`.
    let is_equation = |ts: &[i64], j: usize| -> bool {
        let t = ts[j];
        t - t_min >= tl1 && j > lags && ts[j - lags - 1] == t - lags_i - 1
    };
    // Positions of the entity's instrument rows for the equation at time t.
    let instruments = |ts: &[i64], t: i64| -> Range<usize> {
        let lo = ts.partition_point(|&s| s < t.saturating_sub(max_lag));
        let hi = ts.partition_point(|&s| s <= t - min_lag);
        lo..hi.max(lo)
    };
    let key = |t: i64, s: i64| if collapse { (t - s, 0) } else { (t, s) };

    // First pass: count the equations and the entities that have one, and bound the
    // instrument matrix before collecting its columns.
    let mut n_eq = 0usize;
    let mut n_blocks = 0usize;
    let mut widest = 0usize;
    for e in 0..n_levels {
        resource::checkpoint()?;
        let ts = &st[start[e]..start[e + 1]];
        let before = n_eq;
        for j in 0..ts.len() {
            if is_equation(ts, j) {
                n_eq += 1;
                widest = widest.max(instruments(ts, ts[j]).len());
            }
        }
        if n_eq > before {
            n_blocks += 1;
        }
    }
    if n_eq == 0 {
        let mut distinct: Vec<i64> = resource::try_with_capacity(n, "GMM distinct times")?;
        distinct.extend_from_slice(times);
        distinct.sort_unstable();
        distinct.dedup();
        let g = distinct.windows(2).fold(0, |g, w| gcd(g, w[1] - w[0]));
        let hint = if g > 1 {
            format!(
                "; the values of '{time}' are all multiples of {g} apart; recode them as \
                 consecutive period numbers"
            )
        } else {
            String::new()
        };
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: format!(
                "difference GMM formed no equation: the equation of period t needs '{y}' \
                 observed at t, t-1, ..., t-{} of one entity, and lags are steps of 1 in \
                 '{time}'{hint}",
                lags + 1
            ),
        });
    }
    if n_blocks < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: n_blocks,
            context: "GMM requires at least 2 entities with a differenced equation".to_string(),
        });
    }
    if n_eq < k + 1 {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: n_eq,
            context: "GMM estimation".to_string(),
        });
    }
    let what = "GMM instrument matrix";
    resource::check_alloc::<f64>(
        resource::checked_len(&[n_eq, widest.saturating_add(k_x)], what)?,
        what,
    )?;

    // Second pass: the instrument columns some equation observes, in ascending order.
    let mut keys = BTreeSet::new();
    for e in 0..n_levels {
        resource::checkpoint()?;
        let ts = &st[start[e]..start[e + 1]];
        for j in 0..ts.len() {
            if is_equation(ts, j) {
                let t = ts[j];
                for &s in &ts[instruments(ts, t)] {
                    keys.insert(key(t, s));
                }
            }
        }
    }
    let keys: Vec<(i64, i64)> = keys.into_iter().collect();
    let n_gmm = keys.len();
    let n_inst = n_gmm + k_x;
    if n_inst < k {
        return Err(EconError::InvalidSpecification {
            message: format!("Underidentified: {n_inst} instruments for {k} parameters"),
        });
    }

    // Third pass: the equations, entity block by entity block.
    let mut dy = resource::try_vec(n_eq, 0.0, "GMM differenced outcome")?;
    let mut xm = resource::try_zeros2(n_eq, k, "GMM regressor matrix")?;
    let mut z = resource::try_zeros2(n_eq, n_inst, what)?;
    let mut eq_time = resource::try_vec(n_eq, 0i64, "GMM equation times")?;
    let mut blocks = resource::try_with_capacity(n_blocks, "GMM entity blocks")?;
    let mut block_labels = resource::try_with_capacity(n_blocks, "GMM entity blocks")?;
    let mut entities_without_equations = Vec::new();
    let mut r = 0usize;
    for e in 0..n_levels {
        resource::checkpoint()?;
        let base = start[e];
        let ts = &st[base..start[e + 1]];
        let row = |p: usize| order[base + p];
        let first = r;
        for j in 0..ts.len() {
            if !is_equation(ts, j) {
                continue;
            }
            let t = ts[j];
            let (now, prev) = (row(j), row(j - 1));
            dy[r] = yv[now] - yv[prev];
            for l in 1..=lags {
                xm[[r, l - 1]] = yv[row(j - l)] - yv[row(j - l - 1)];
            }
            for (m, xc) in xv.iter().enumerate() {
                let d = xc[now] - xc[prev];
                xm[[r, lags + m]] = d;
                z[[r, n_gmm + m]] = d;
            }
            for p in instruments(ts, t) {
                if let Ok(c) = keys.binary_search(&key(t, ts[p])) {
                    z[[r, c]] = yv[row(p)];
                }
            }
            eq_time[r] = t;
            r += 1;
        }
        if r > first {
            blocks.push(first..r);
            block_labels.push(factor.levels[e].clone());
        } else {
            entities_without_equations.push(factor.levels[e].clone());
        }
    }

    Ok(DiffEquations {
        dy,
        x: xm,
        z,
        n_gmm_instruments: n_gmm,
        time: eq_time,
        blocks,
        block_labels,
        entities_without_equations,
        span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use crate::data::columns::{ColumnRequest, NaPolicy, numeric_column};
    use polars::prelude::Column;

    /// The case's dataset (`panel_grunfeld_na`) and the complete Grunfeld panel it was
    /// built from.
    fn fixtures(case_id: &str) -> (Dataset, Dataset) {
        let (case, na) = crate::test_support::load_case(case_id);
        let path = p2a_validation::repo_path(&case.datasets[0]);
        assert!(
            path.ends_with("validation/datasets/grunfeld.csv"),
            "{path:?}"
        );
        (na, Dataset::new(DataLoader::load_csv(&path).unwrap()))
    }

    /// `ds` with column `name` replaced by `column`.
    fn with(ds: &Dataset, column: Column) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(column).unwrap();
        Dataset::new(df)
    }

    /// A committed CSV loaded as `p2a data load` loads it; for tests that compare no R
    /// value, so they load no case.
    fn repo_dataset(rel: &str) -> Dataset {
        Dataset::new(DataLoader::load_csv(p2a_validation::repo_path(rel)).unwrap())
    }

    const AB_BALANCED: &str = "validation/datasets/generated/ab_balanced.csv";

    fn lag_spec(lags: usize, collapse: bool) -> GmmLagSpec {
        GmmLagSpec {
            lags,
            min_lag: 2,
            max_lag: None,
            collapse,
        }
    }

    /// The equations of `y` on `x` with entity `id` and time `year`.
    fn ab_equations(ds: &Dataset, spec: &GmmLagSpec) -> EconResult<DiffEquations> {
        let cols = ColumnRequest::new(ds.df())
            .factor("id")
            .time_index("year")
            .numeric("y")
            .numeric("x")
            .na(NaPolicy::Omit)
            .extract()?;
        difference_equations(&cols, "y", &["x"], "id", "year", spec)
    }

    /// pgmm's layout on a panel with interior gaps and late entry: an equation needs
    /// y at t, t-1 and t-2; the instruments of an equation are the entity's observed
    /// levels two or more periods back, one column per (period, lag) or per lag depth.
    #[test]
    fn test_validate_difference_equations_gaps_layout() {
        let (case, ds) = crate::test_support::load_case("dpd_ab_gaps");
        let eq = ab_equations(&ds, &lag_spec(1, false)).unwrap();
        let mut check = p2a_validation::Check::new(&case);
        let exact = p2a_validation::Tol::exact();
        check
            .scalar("onestep_n_obs", eq.dy.len() as f64, exact)
            .scalar("onestep_n_instruments_nonzero", eq.z.ncols() as f64, exact);
        let n_eq = eq.dy.len();
        assert_eq!(eq.x.dim(), (n_eq, 2));
        assert_eq!(eq.z.nrows(), n_eq);
        assert_eq!(eq.time.len(), n_eq);
        assert_eq!(eq.n_gmm_instruments, 21);
        assert_eq!(eq.blocks.len(), 100);
        assert_eq!(eq.span, 8);
        assert!(eq.entities_without_equations.is_empty());
        let times = |b: usize| eq.time[eq.blocks[b].clone()].to_vec();
        assert_eq!(eq.block_labels[8], "9");
        assert_eq!(times(8), [2002, 2003, 2007]);
        assert_eq!(eq.block_labels[12], "13");
        assert_eq!(times(12), [2004, 2005, 2006, 2007]);

        // id 9 (no 2004) read from the case CSV: year -> (y, x).
        let column = |name: &str| numeric_column(ds.df(), name, NaPolicy::Fail).unwrap().0;
        let (ids, years, y, x) = (column("id"), column("year"), column("y"), column("x"));
        let id9: std::collections::BTreeMap<i64, (f64, f64)> = (0..ids.len())
            .filter(|&i| ids[i] == 9.0)
            .map(|i| (years[i] as i64, (y[i], x[i])))
            .collect();
        let (y_at, x_at) = (|t: i64| id9[&t].0, |t: i64| id9[&t].1);
        let r = eq.blocks[8].start + 2;
        assert_eq!(eq.time[r], 2007);
        let mut gmm: Vec<f64> =
            eq.z.row(r)
                .iter()
                .take(eq.n_gmm_instruments)
                .copied()
                .filter(|v| *v != 0.0)
                .collect();
        gmm.sort_by(f64::total_cmp);
        let mut expected: Vec<f64> = [2000, 2001, 2002, 2003, 2005].map(y_at).to_vec();
        expected.sort_by(f64::total_cmp);
        assert_eq!(gmm, expected, "GMM instruments of (id 9, 2007)");
        assert_eq!(eq.z[[r, 21]], x_at(2007) - x_at(2006));
        assert_eq!(eq.dy[r], y_at(2007) - y_at(2006));
        assert_eq!(
            eq.x.row(r).to_vec(),
            [y_at(2006) - y_at(2005), x_at(2007) - x_at(2006)]
        );

        let collapsed = ab_equations(&ds, &lag_spec(1, true)).unwrap();
        check.scalar(
            "collapse_twostep_n_instruments_nonzero",
            collapsed.z.ncols() as f64,
            exact,
        );
        let lags2 = ab_equations(&ds, &lag_spec(2, false)).unwrap();
        check.scalar("lags2_twostep_n_obs", lags2.dy.len() as f64, exact);
        check.finish();
    }

    #[test]
    fn test_difference_equations_duplicate_is_an_error() {
        let ds = repo_dataset(AB_BALANCED);
        let mut df = ds.df().clone();
        df.vstack_mut(&ds.df().slice(2, 1)).unwrap();
        match ab_equations(&Dataset::new(df), &lag_spec(1, false)) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("Entity '1' has 2 rows for time 2002"),
                "{message}"
            ),
            Err(e) => panic!("expected InvalidSpecification, got {e:?}"),
            Ok(_) => panic!("expected InvalidSpecification, got equations"),
        }
    }

    /// The balanced panel's first `periods` years.
    fn first_periods(periods: i64) -> Dataset {
        let ds = repo_dataset(AB_BALANCED);
        let keep: polars::prelude::BooleanChunked = ds
            .df()
            .column("year")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|y| y.map(|y| y < 2000 + periods))
            .collect();
        Dataset::new(ds.df().filter(&keep).unwrap())
    }

    /// pgmm fits a panel of lags + 2 periods: the last period's equation, with the
    /// levels lags + 1 and more periods back as its instruments. One period fewer is
    /// refused, and the error names lags + 2.
    #[test]
    fn test_difference_equations_span_of_lags_plus_two() {
        for lags in [1usize, 2] {
            let span = lags + 2;
            let eq = ab_equations(&first_periods(span as i64), &lag_spec(lags, false)).unwrap();
            assert_eq!(eq.span, span);
            assert_eq!(eq.dy.len(), 100, "one equation per entity");
            assert!(eq.time.iter().all(|&t| t == 2000 + span as i64 - 1));
            assert_eq!(eq.n_gmm_instruments, lags);
            match ab_equations(&first_periods(span as i64 - 1), &lag_spec(lags, false)) {
                Err(EconError::InsufficientData {
                    required,
                    provided,
                    context,
                }) => {
                    assert_eq!((required, provided), (lags + 2, lags + 1), "{context}");
                    assert!(
                        context.contains(&format!(
                            "at least {} time periods for {lags} lags",
                            lags + 2
                        )),
                        "{context}"
                    );
                }
                Err(e) => panic!("expected InsufficientData, got {e:?}"),
                Ok(_) => panic!("expected InsufficientData, got equations"),
            }
        }
    }

    /// Lags are steps of one period: biennial years form no equation, and the error
    /// says how to recode them.
    #[test]
    fn test_difference_equations_need_steps_of_one() {
        let ds = repo_dataset(AB_BALANCED);
        let years: Vec<i64> = ds
            .df()
            .column("year")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|y| 2000 + 2 * (y.unwrap() - 2000))
            .collect();
        let biennial = with(&ds, Column::new("year".into(), years));
        match ab_equations(&biennial, &lag_spec(1, false)) {
            Err(EconError::InsufficientData { context, .. }) => assert!(
                context.contains("consecutive period numbers")
                    && context.contains("multiples of 2"),
                "{context}"
            ),
            Err(e) => panic!("expected InsufficientData, got {e:?}"),
            Ok(_) => panic!("expected InsufficientData, got equations"),
        }
    }

    /// The crate-internal codes: dense period indexes and factor codes as `usize`.
    #[test]
    fn test_validate_na_entity_and_time_codes() {
        assert_eq!(
            time_codes(&[2001, 1999, 2001, 2000]),
            (vec![2, 0, 2, 1], vec![1999, 2000, 2001])
        );
        let (na, _) = fixtures("panel_grunfeld_na");
        let cols = ColumnRequest::new(na.df())
            .factor("firm_id")
            .extract()
            .unwrap();
        let (codes, n_groups) = entity_codes(cols.factor("firm_id"));
        assert_eq!((codes.len(), n_groups), (198, 10));
        let ids = numeric_column(na.df(), "firm_id", NaPolicy::Omit)
            .unwrap()
            .0;
        assert_eq!(ids.len(), 198);
        for (code, id) in codes.iter().zip(&ids) {
            assert_eq!(*code, *id as usize, "firm_id {id} has code {code}");
        }
    }
}
