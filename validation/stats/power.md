# Validation: Power Analysis (power.t.test, power.prop.test, power.anova.test)

## Method Overview

`crates/p2a-core/src/stats/power.rs` computes the power of three tests, or solves for any one
other parameter, as R's `stats` package does:

| p2a | R | Parameters (any one can be the unknown) |
|---|---|---|
| `power_t_test_with(&PowerTTestSpec)` | `power.t.test` | `n`, `delta`, `sd`, `sig_level`, `power` |
| `power_prop_test_with(&PowerPropTestSpec)` | `power.prop.test` | `n`, `p1`, `p2`, `sig_level`, `power` |
| `power_anova_test_with(&PowerAnovaTestSpec)` | `power.anova.test` | `groups`, `n`, `between_var`, `within_var`, `sig_level`, `power` |

The Tier 0 functions `power_t_test`, `power_prop_test`, `power_anova_test` and the `run_power_*`
wrappers keep their signatures and call the `_with` functions with `strict = false` and R's default
tolerance. `PowerAnovaTestResult::groups` is now `f64`: a solved group count is fractional, as R's.

Status: **R-pinned** (`validation/methods/stats_power_loglin.json`, entries `power_t_test`,
`power_prop_test`, `power_anova_test`, `noncentral_distributions`).

## Parameters and defaults

- `sd = 1` and `sig_level = 0.05` unless they are the unknown (R's defaults).
- `strict = false` (R's default): a two-sided test counts only the tail in the direction of the
  effect. `strict = true` adds the other tail.
- A two-sided `delta` is replaced by its absolute value (R). A one-sided test in the `_with` function
  tests the upper tail whatever the sign of `delta` (R).
- `tol` is the root-search tolerance, default `R_UNIROOT_TOL = 2^-13` (R's
  `.Machine$double.eps^0.25`). `power.anova.test` has no `tol` argument in R; the spec has one.
- Results report `solved_for`, the root search (`root: Option<RootInfo>`: iterations, extension
  steps, final bracket and width, convergence), `warnings`, and for ANOVA Cohen's
  `effect_size_f = sqrt((groups - 1) / groups * between_var / within_var)` (R's `between.var` is the
  variance of the group means, divisor `groups - 1`).
- `Display` follows R's `print.power.htest`: the method line, `name = value` rows with 7 significant
  digits, `NOTE:` and then the warnings.

## Which parameter is solved

1. `solve_for` when given. That parameter must be absent, and every parameter without a default
   must be given ("solve_for = sd but sd = 1.2 was given; omit it", "n is required when solving for
   sd").
2. Otherwise the only absent parameter, defaults included: `{n, delta, power}` given with `sd`
   absent solves for `sd`.
3. Otherwise `sd` and `sig_level` take their defaults and exactly one of the parameters without a
   default must be absent ("exactly one of n, delta, power must be omitted ...; omitted: n, delta").

R reports an over-specified call where rule 2 solves for an absent `sd` or `sig_level` (in R they
are never `NULL` unless passed as `NULL`); p2a solves it, since the absent value says which one is
wanted.

## Search intervals (R's)

| Function | Unknown | Interval | Extended |
|---|---|---|---|
| t | `n` | [2, 1e7] | upwards crossing (`upX`) |
| t | `sd` | `delta * [1e-7, 1e7]` | `downX` |
| t | `delta` | `sd * [1e-7, 1e7]` | `upX` |
| t | `sig_level` | [1e-10, 1 - 1e-10] | both ends (`yes`) |
| prop | `n` | [1, 1e7] | `upX` |
| prop | `p1` | [0, p2] | `yes` |
| prop | `p2` | [p1, 1] | `yes` |
| prop | `sig_level` | [1e-10, 1 - 1e-10] | `upX` |
| anova | `groups` | [2, 100] | no |
| anova | `n` | [2, 1e5] | no |
| anova | `within_var` | `between_var * [1e-7, 1e7]` | no |
| anova | `between_var` | `within_var * [1e-7, 1e7]` | no |
| anova | `sig_level` | [1e-10, 1 - 1e-10] | no |

An extension step moves an end by `0.01 * max(1e-4, |end|)`, doubling each step, at most 1000
steps. A power that does not reach the target on the (extended) interval is
`InvalidSpecification("cannot solve for <p>: ...")` (R's uniroot error). Solving `sd` needs
`delta > 0`, and solving `within_var` needs `between_var > 0` (R's interval is empty otherwise).
The solved t-test `n` may lie below 2 (R: `power.t.test(delta = 10, power = 0.8)$n = 1.67`).

## Algorithms

- **Noncentral t**: Lenth (1989), Algorithm AS 243, written from the paper with R's constants (error
  bound 1e-12, 1000 terms). For df > 4e5, or a noncentrality whose Poisson weights underflow
  (`ncp^2 > 2 * 1021 * ln 2`), the normal approximation of Abramowitz & Stegun 26.7.10.
- **Noncentral beta and F**: Lenth (1987), Algorithm AS 226, with Frick's (1990) AS R84 start
  `x0 = floor(max(0, c - 7 sqrt(c)))`, `c = ncp / 2`, error bound 1e-9, 10000 terms. For df2 > 1e8
  the noncentral chi-square limit: the Poisson mixture of central chi-squares from the same start
  (error bound 1e-12, 10000 terms), with the cube-root normal approximation of A&S 26.4.32 beyond
  `ncp / 2 > 1e5` or `df > 1e10`.
- **Critical values**: `qt` is `traits::estimator::t_quantile` at the exact tail
  `sig_level / tside`; `qnorm` is `normal_quantile`; the F upper critical value is the root of
  `ln f_test_p_value(q) = ln alpha` by Brent's zeroin at tolerance 0, with R's chi-square limits when
  a df exceeds 4e5 (Wilson-Hilferty beyond 1e10). No tail is formed as `1 - sig_level`.
- **Root finder**: Brent's (1973) zeroin, with R's search intervals and bracket-extension rule
  (`extendInt`). `RootInfo` is p2a's own diagnostic record, not R's iteration count.
- Every power evaluation calls `resource::checkpoint()`; every loop is bounded.

## Precision warnings

R's nmath prints "full precision may not have been achieved" warnings during its root searches,
including at intermediate values of a search (one-sided `sig.level` solves warn in `pnt`). p2a
evaluates the series once more at the returned value and reports a result warning only when that
value is imprecise. The proportion test reports R's three warnings verbatim ("No p1 in [0, p2] can be
found to achieve the desired power", and the `p2` and significance-level forms).

## R reference case and tests

Case `stat_power_solve` (`validation/r/cases/stats_power_loglin.R`): every parameter solved for
every type and alternative at `tol = 1e-12`, the strict variants, the default-tolerance roots
(`*_pkg`), the extended-bracket root `t_extend_n`, and one point per branch of `pt`/`pf` with a
noncentrality.

| Test (`p2a-core`, `stats::power::tests::`) | What |
|---|---|
| `test_validate_noncentral_t_f_ports` | `pt_value`, `pf_value` at `Tol::new(1e-12, 1e-14)` |
| `test_validate_power_legacy_api` | the Tier 0 functions at R's defaults and default tolerance |
| `test_validate_power_t_solve_power`, `_n`, `_delta`, `_sd`, `_sig` | the t test, every unknown, type and alternative |
| `test_validate_power_prop_solve` | the proportion test, every unknown and alternative |
| `test_validate_power_anova_solve` | ANOVA, every unknown, tight and default tolerance |

Tolerances:

- `POWER_TOL = Tol::new(1e-10, 1e-14)`: direct power. The series sum the same terms as R; the
  quantile helpers differ from R's `qt`/`qnorm` by a few 1e-13 relative.
- `ROOT_TOL = Tol::new(1e-9, 1e-12)`: roots at `tol = 1e-12` on both sides.
- Default-tolerance roots: against R's `*_pkg` at `Tol::new(8 eps, 2 tol)` and against R's tight
  root at `Tol::new(4 eps, tol)`, `tol = 2^-13`: Brent's zeroin leaves `|x - x*| <= tol + 4 eps |x|`,
  and R's value obeys the same bound.

The tests without a case (inference rule, validation messages, option spellings, Display, the
cancelled budget, extreme inputs, the critical values against R's `qt`/`qf`) are listed in the
module.

## Known differences

- Precision warnings at intermediate values of a root search are not reported (above).
- An absent `sd` or `sig_level` that is the only absent parameter is solved (above).
- `power_t_test` (the Tier 0 function) runs a one-sided test in the direction of `delta`, so a
  negative `delta` has the power of `|delta|`; R and `power_t_test_with` test the upper tail
  (power 1.0e-6 at `n = 20, delta = -1`). Registry deviation `STAT-61`.
- Beyond `df2 = 1e8` R calls its own `pnchisq`; p2a sums the same Poisson mixture with its own error
  bound (`|difference| < 1e-14` on the pinned point) and, far out, uses the A&S normal approximation
  where R's series is slow or does not converge (R warns there).
- `power.anova.test(groups = 2^62, power = 0.8)` fails in R and here ("cannot solve for n"): the
  power is 1 on the whole interval.
