//! Ranking helpers shared by the rank-based tests of [`crate::stats`].
//!
//! [`midranks()`] is R's `rank(x)` (`ties.method = "average"`). Values are ordered with
//! `f64::total_cmp` and two values tie only when they compare equal (`==`), so `-0.0` and `0.0`
//! tie and two distinct doubles never share a rank, whatever their magnitude. kruskal.test,
//! friedman.test, quade.test, fligner.test, mood.test, ansari.test and wilcox.test all rank this
//! way; the earlier copies that treated |a - b| < 1e-10 as a tie gave H = 0, p = 1 on data
//! recorded in SI base units (STAT-53).

use crate::errors::{EconError, EconResult};
use crate::resource::{try_vec, try_with_capacity};

/// Midranks of `x` in input order (1-based) and the size of every tie group, one entry per
/// distinct value in ascending order of the value, singletons included, so the sizes sum to
/// `x.len()` (R: `as.vector(table(x))`). NaN is refused (`InvalidSpecification` naming the
/// 0-based position); `±Inf` rank as the extremes; an empty slice gives two empty vectors.
///
/// ```
/// use p2a_core::stats::ranks::midranks;
/// let (ranks, ties) = midranks(&[3e-12, 1e-12, 3e-12]).unwrap();
/// assert_eq!(ranks, vec![2.5, 1.0, 2.5]);
/// assert_eq!(ties, vec![1, 2]);
/// ```
pub fn midranks(x: &[f64]) -> EconResult<(Vec<f64>, Vec<usize>)> {
    let n = x.len();
    let mut pairs: Vec<(f64, usize)> = try_with_capacity(n, "midranks order")?;
    let mut ranks: Vec<f64> = try_vec(n, 0.0, "midranks")?;
    let mut sizes: Vec<usize> = Vec::new();
    midranks_into(x, &mut pairs, &mut ranks, &mut sizes)?;
    Ok((ranks, sizes))
}

/// [`midranks()`] into caller-owned buffers, for callers that rank many short vectors (one per
/// block): the three buffers are cleared and refilled, so a loop allocates nothing once they
/// have grown to the longest input. `ranks` holds the midranks of `x` in input order and `sizes`
/// the tie-group sizes in ascending order of the value; `pairs` is scratch. Sorting
/// `(value, index)` pairs keeps each comparison on contiguous memory (cheaper than comparing
/// through an index vector).
pub(crate) fn midranks_into(
    x: &[f64],
    pairs: &mut Vec<(f64, usize)>,
    ranks: &mut Vec<f64>,
    sizes: &mut Vec<usize>,
) -> EconResult<()> {
    if let Some(i) = x.iter().position(|v| v.is_nan()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "midranks: value {i} (0-based) is NaN; remove missing values before ranking"
            ),
        });
    }
    let n = x.len();
    pairs.clear();
    pairs.extend(x.iter().copied().zip(0..n));
    // Ties are averaged, so the sort need not be stable; under total_cmp every run of values
    // equal by `==` (-0.0 and 0.0 included) is contiguous.
    pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    ranks.clear();
    ranks.resize(n, 0.0);
    sizes.clear();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        let rank = (i + 1 + j) as f64 / 2.0;
        for &(_, k) in &pairs[i..j] {
            ranks[k] = rank;
        }
        sizes.push(j - i);
        i = j;
    }
    Ok(())
}

/// Σ (t³ − t) over tie-group sizes: the tie term of Kruskal-Wallis, Friedman and Wilcoxon (0
/// without ties).
///
/// ```
/// use p2a_core::stats::ranks::tie_cube_sum;
/// assert_eq!(tie_cube_sum(&[1, 2, 3]), 30.0);
/// ```
pub fn tie_cube_sum(sizes: &[usize]) -> f64 {
    sizes
        .iter()
        .map(|&t| {
            let t = t as f64;
            t * t * t - t
        })
        .sum()
}

/// Selection helpers for the long-format case datasets of the rank-test pins.
#[cfg(test)]
pub(crate) mod test_data {
    use crate::data::Dataset;
    use crate::data::columns::{ColumnRequest, Columns, NaPolicy};

    /// The extracted columns (`extra` factors and `value` numeric, plus every filter column) and
    /// the rows whose filter columns carry the given labels.
    fn matching_rows(
        ds: &Dataset,
        value: Option<&str>,
        extra: Option<&str>,
        filters: &[(&str, &str)],
    ) -> (Vec<usize>, Columns) {
        let mut factors: Vec<&str> = filters.iter().map(|(c, _)| *c).collect();
        factors.extend(extra);
        let mut req = ColumnRequest::new(ds.df()).factors(&factors);
        if let Some(v) = value {
            req = req.numeric(v);
        }
        let cols = req.na(NaPolicy::Fail).extract().unwrap();
        let rows = (0..cols.n())
            .filter(|&i| filters.iter().all(|(c, l)| cols.factor(c).label(i) == *l))
            .collect();
        (rows, cols)
    }

    /// The numeric column `value` of the rows whose factor columns carry the given labels, in
    /// row order (the long-format case datasets). Missing values are an error (`NaPolicy::Fail`).
    pub(crate) fn column_where(ds: &Dataset, value: &str, filters: &[(&str, &str)]) -> Vec<f64> {
        let (rows, cols) = matching_rows(ds, Some(value), None, filters);
        let v = cols.numeric(value);
        rows.into_iter().map(|i| v[i]).collect()
    }

    /// The labels of factor `column` that occur among the rows matching `filters`, in R factor
    /// order.
    pub(crate) fn levels_where(
        ds: &Dataset,
        column: &str,
        filters: &[(&str, &str)],
    ) -> Vec<String> {
        let (rows, cols) = matching_rows(ds, None, Some(column), filters);
        let f = cols.factor(column);
        let mut present = vec![false; f.levels.len()];
        for i in rows {
            present[f.codes[i] as usize] = true;
        }
        f.levels
            .iter()
            .zip(present)
            .filter(|(_, p)| *p)
            .map(|(l, _)| l.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_midranks_exact_ties_small_scale() {
        // STAT-53: R rank() ties only equal doubles. Every value of stat_rank_small_groups lies
        // within 1e-10 of every other; the old Kruskal/Friedman helpers made them one tie group.
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_small_groups");
        let col = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
        let (r, sizes) = midranks(&col("y")).unwrap();
        let (ru, sizes_u) = midranks(&col("y_unit")).unwrap();
        assert_eq!(
            (&r, &sizes),
            (&ru, &sizes_u),
            "ranks do not depend on the unit"
        );
        let (rnt, snt) = midranks(&col("ynt")).unwrap();
        assert!(snt.iter().all(|&t| t == 1));
        let sizes_f: Vec<f64> = sizes.iter().map(|&t| t as f64).collect();
        Check::new(&case)
            .vector("midranks_y", &r, Tol::exact())
            .vector("tie_sizes_y", &sizes_f, Tol::exact())
            .vector("midranks_ynt", &rnt, Tol::exact())
            .finish();
        // R: rank(c(3e-12, 1e-12, 2e-12, 3e-12, 2.5e-12)) = 4.5 1 2 4.5 3 (a 1e-10 tolerance
        // gives 3 3 3 3 3)
        assert_eq!(
            midranks(&[3e-12, 1e-12, 2e-12, 3e-12, 2.5e-12]).unwrap(),
            (vec![4.5, 1.0, 2.0, 4.5, 3.0], vec![1, 1, 1, 2])
        );
    }

    #[test]
    fn test_midranks_signed_zero_infinities_and_nan() {
        // R: rank(c(0, -0, 1)) = 1.5 1.5 3; rank(c(Inf, -Inf, 2, Inf)) = 3.5 1 2 3.5, table
        // sizes 1 1 2
        assert_eq!(
            midranks(&[0.0, -0.0, 1.0]).unwrap(),
            (vec![1.5, 1.5, 3.0], vec![2, 1])
        );
        assert_eq!(
            midranks(&[f64::INFINITY, f64::NEG_INFINITY, 2.0, f64::INFINITY]).unwrap(),
            (vec![3.5, 1.0, 2.0, 3.5], vec![1, 1, 2])
        );
        assert_eq!(midranks(&[]).unwrap(), (vec![], vec![]));
        match midranks(&[1.0, f64::NAN, 2.0]) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.contains("value 1") && message.contains("NaN"),
                    "{message}"
                )
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_midranks_into_reuses_buffers() {
        let (mut pairs, mut ranks, mut sizes) = (Vec::new(), Vec::new(), Vec::new());
        let long = [4.0, 1.0, 4.0, 2.0, 0.5, 4.0];
        let short = [2.0, 2.0, 1.0];
        midranks_into(&long, &mut pairs, &mut ranks, &mut sizes).unwrap();
        assert_eq!((ranks.clone(), sizes.clone()), midranks(&long).unwrap());
        midranks_into(&short, &mut pairs, &mut ranks, &mut sizes).unwrap();
        assert_eq!((ranks, sizes), midranks(&short).unwrap());
    }

    #[test]
    fn test_tie_cube_sum() {
        assert_eq!(tie_cube_sum(&[1, 2, 3]), 30.0); // 0 + 6 + 24
        assert_eq!(tie_cube_sum(&[1, 1]), 0.0);
        assert_eq!(tie_cube_sum(&[]), 0.0);
    }
}
