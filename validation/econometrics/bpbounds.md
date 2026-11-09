# Balke-Pearl bounds

**File:** `crates/p2a-core/src/econometrics/bpbounds.rs` (`run_bp_bounds`, `bp_bounds_from_probs`).

**Status**: not R-pinned (closed-form Balke-Pearl bounds pinned to a hand-computed table, checked against the exact
response-type LP; the R package bpbounds is not installed). Both tables below were checked during planning against
the exact response-type LP (`lpSolve` 5.6.23: 16 response types without monotonicity, 12 without defiers; agreement
5.5e-17).

## Formulas

Cell probabilities `p_{dy|z} = P(D = d, Y = y | Z = z)`.

**Without monotonicity** (Balke & Pearl 1997; Pearl 2009, eq. 8.14-8.15):

```
ACE_lower = max( p11|1 + p00|0 - 1,
                 p11|0 + p00|1 - 1,
                 p11|0 - p11|1 - p01|1 - p10|0 - p01|0,
                 p11|1 - p11|0 - p01|0 - p10|1 - p01|1,
                 -p10|1 - p01|1,
                 -p10|0 - p01|0,
                 p00|1 - p10|1 - p01|1 - p10|0 - p00|0,
                 p00|0 - p10|0 - p01|0 - p10|1 - p00|1 )

ACE_upper = min( 1 - p10|1 - p01|0,
                 1 - p10|0 - p01|1,
                 -p10|0 + p10|1 + p00|1 + p11|0 + p00|0,
                 -p10|1 + p11|1 + p00|1 + p10|0 + p00|0,
                 p11|1 + p00|1,
                 p11|0 + p00|0,
                 -p01|1 + p11|1 + p00|1 + p11|0 + p01|0,
                 -p01|0 + p11|0 + p00|0 + p11|1 + p01|1 )
```

**With monotonicity** (no defiers): `ACE_lower = p11|1 + p00|0 - 1`, `ACE_upper = 1 - p10|1 - p01|0`.

**Wald:** `(P(Y=1|Z=1) - P(Y=1|Z=0)) / (P(D=1|Z=1) - P(D=1|Z=0))`, SE by the delta method with the within-arm
covariance `Cov = (p11|0 - P(Y=1|Z=0) P(D=1|Z=0)) / n_0 + (p11|1 - P(Y=1|Z=1) P(D=1|Z=1)) / n_1`:
`Var = Var(num)/den^2 + num^2 Var(den)/den^4 - 2 num Cov / den^3` (equals 2SLS with HC0 on the expanded data).

**Instrumental inequality** (Pearl 1995): for each `d`, the result warns when
`sum_y max_z P(Y = y, D = d | Z = z) > 1` ("the instrumental inequality fails (d = <d>: <sum> > 1): the exclusion
restriction or randomization of Z is violated; the Balke-Pearl bounds are not valid"), with and without
monotonicity and on both entry points. The check does not depend on whether the bounds cross: a table can violate
the inequality with lower <= upper.

**Bootstrap:** units resampled (seeded; `seed = None` uses 42), type-7 percentile intervals (R `quantile(type =
7)`); fewer than 100 valid replicates give no interval.

## Tables

| Table | Counts (Z=0: 00, 01, 10, 11 / Z=1: 00, 01, 10, 11) | Bounds (no mono) | Monotone bounds | Wald | Wald SE |
|---|---|---|---|---|---|
| Vitamin A (Sommer et al. 1986; published [-0.1946, 0.0054]) | 74, 11514, 0, 0 / 34, 2385, 12, 9663 | -0.1946228482113902, 0.0053936889144572397 | same | 0.0032280386285733001 | 0.0011591629284325984 |
| Table 2 (40 + 40) | 12, 10, 8, 10 / 6, 4, 10, 20 | -0.2, 0.5 | -0.2, 0.5 | 1/3 | 0.36830442319927159 |

Table 2 by hand: p = 0.30, 0.25, 0.20, 0.25 | 0.15, 0.10, 0.25, 0.50.
- Lower terms: -0.20, -0.60, -0.80, -0.35, -0.35, -0.45, -0.70, -0.55; max -0.20.
- Upper terms: 0.50, 0.70, 0.75, 0.90, 0.65, 0.55, 1.05, 0.90; min 0.50.
- Monotone: lower `0.50 + 0.30 - 1 = -0.20`, upper `1 - 0.25 - 0.25 = 0.50`.
- Wald `(0.60 - 0.50) / (0.75 - 0.45) = 1/3`. `Var(num) = (0.25 + 0.24)/40`, `Var(den) = (0.2475 + 0.1875)/40`,
  `Cov = (0.025 + 0.05)/40`, so `Var = 0.01225/0.09 + 0.01 * 0.010875/0.0081 - 2 * 0.1 * 0.001875/0.027 =
  0.1356481...`, SE 0.368304... (0.386700... without the covariance term).

## Rust Test Functions

`crates/p2a-core/src/econometrics/bpbounds.rs`:
- `econometrics::bpbounds::tests::test_bpbounds_closed_form_tables`: both tables, every value above to 1e-12.
- `econometrics::bpbounds::tests::test_bp_bounds_percentile_rule`: type-7 interval of 0.1, ..., 1.0 is
  [0.1225, 0.9775].
- `econometrics::bpbounds::tests::test_bp_bounds_bootstrap_ci_seeded`: no seed equals seed 42; the intervals are
  those of the replayed replicates.
- `econometrics::bpbounds::tests::test_perfect_compliance`: with perfect compliance the Wald estimate is the ITT.
- `econometrics::bpbounds::tests::test_bpbounds_inequality_fails_without_crossing`: a table with uncrossed bounds
  [0.2, 0.6] that violates the inequality (d = 1: 1.2 > 1) warns, in both monotonicity modes and on both paths.

## References

- Balke, A., & Pearl, J. (1997). Bounds on treatment effects from studies with imperfect compliance. *JASA*,
  92(439), 1171-1176.
- Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University Press, chapter 8.
- Sommer, A., Tarwotjo, I., Djunaedi, E., et al. (1986). Impact of vitamin A supplementation on childhood
  mortality. *The Lancet*, 327(8491), 1169-1173.
- Palmer, T. M., Ramsahai, R. R., Didelez, V., & Sheehan, N. A. (2011). Nonparametric bounds for the causal
  effect in a binary instrumental-variable model. *Stata Journal*, 11(3), 345-367.
