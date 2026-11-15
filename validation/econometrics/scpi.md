# Synthetic Control with Prediction Intervals (SCPI) Validation

**Status**: R-pinned, known deviations (CAUS-32) (registry `validation/methods/causal_gsynth.json`, id
`causal_scpi`): the simplex intervals are pinned to R's `scpi` 3.0.1 at its defaults; the deviations
are solver accuracy (scpi's ECOS solves) and the random generator.

## What p2a computes

`run_scpi` (`crates/p2a-core/src/econometrics/scpi.rs`) fits donor weights on the pre-treatment
outcomes and computes the prediction intervals of Cattaneo, Feng and Titiunik (2021) as
`scpi::scpi()` does at its defaults.

- Weights: simplex (`min ||y - X w||^2`, `w >= 0`, `sum w = 1`, the exact simplex QP shared with the
  synthetic control; non-unique optima give the minimum-norm weights and a warning), lasso, ridge or
  lasso-simplex. Penalties must be finite, ridge lambda > 0, lasso lambda >= 0.
- Lasso-simplex (CAUS-81): on the simplex `sum |w| = 1`, so the L1 penalty is a constant and the
  problem is the simplex least squares fit; `lasso_simplex` returns the simplex weights for every
  lambda, with the warning "lasso_simplex: on the simplex the L1 penalty is constant, so lambda has
  no effect; the weights are the simplex weights" (`test_lasso_simplex_equals_simplex`).
- In-sample uncertainty (scpi's `u.missp = TRUE`, `u.sigma = "HC1"`, `u.order = 1`, `u.lags = 0`):
  the pseudo-residuals' conditional mean is the linear fit on the active donors and a constant (a
  constant alone when T0 - 10 does not exceed the columns); Sigma = B' Omega B / T0^2 with the HC1
  weights `T0 / (T0 - df)`, df the number of non-zero weights minus 1. rho is scpi's type-2 rule
  capped at `rho.max = 0.2` (with scpi's fallbacks below 0.001 and 0.05, and scpi's warning when
  the first raises rho); a donor whose weight is
  below rho keeps it as its lower bound. Per simulation G = Sigma^(1/2) z with z standard normal,
  and per post period the minimum and maximum of `p_t' delta` subject to
  `delta' Q delta - 2 G' delta <= 0` (Q = B'B / T0) over the localised simplex. The in-sample bounds
  are the `u_alpha / 2` and `1 - u_alpha / 2` quantiles (type 7) of `-max` and `-min` over the
  simulations. Lasso and ridge use the sets `||w + delta||_1 <= ||w||_1 + rho` and
  `||w + delta||_2 <= ||w||_2 + rho` (not R-pinned).
- The programs are solved by a logarithmic barrier method: one phase I per simulation finds a
  strictly feasible point, every objective of that simulation starts from it, and the duality gap is
  1e-10 of the scaled problem. The gap bound holds only near the central path, so the last centring
  must end with a Newton decrement of at most 1/4 (gap at most `(2m + 1) / t`); a centring that ends
  above it is retried from the last centred point with a shorter step in t. A simulation whose
  program fails, or ends uncentred, is left out of the quantiles and counted (`n_failed_sims`, with a
  warning), as an ECOS failure in scpi. An `e_method` other than `all` wins over the deprecated
  `variance_method` (with a warning); `e_method = all` is also the default, so it cannot be told
  apart from no `e_method`, and `variance_method` is then honoured.
- Out-of-sample uncertainty (scpi's `e.method = "all"`, `e.order = 1`, `e.lags = 0`), each side at
  `e_alpha / 2`: sub-Gaussian (`gaussian`: the fitted mean plus `sqrt(-2 log(alpha))` times the
  smaller of the fitted conditional standard deviation and IQR / 1.34 of the quantile-regression
  residuals), location-scale (`ls`: the fitted mean plus the standardised residuals' sample
  quantiles) and quantile regression (`qreg`: Frisch-Newton quantile regressions at `e_alpha / 2`
  and `1 - e_alpha / 2`, or the sample quantiles when the design is a constant). The design is the
  active donors and a constant, collapsed to the constant as scpi's `detectConstant` and
  `avoidCollin` do.
- Interval per method: the synthetic prediction plus the in-sample plus the out-of-sample bounds
  (scpi's `CI.all.<method>`); the effect interval is the actual outcome minus it. With
  `e_method = all` every method is reported and the sub-Gaussian one is the headline.
- Options: `u_alpha`, `e_alpha` (default 0.05 each, so coverage at least 90%); `alpha` is the former
  single level, kept as an alias that sets both to `alpha / 2` (coverage at least `1 - alpha`);
  `sims` (default 200, at least 10); `seed` (default 42, a seeded `StdRng`); `e_method`; `u_missp`;
  `variance_method` is a deprecated alias of `e_method` (`gaussian` -> gaussian, `subgaussian` -> all,
  `loo_cv` and `kfold_cv` -> all with a warning). The previous heuristic interval
  (`effect +/- qt(1 - alpha/2, T0 - 1) sqrt(in + out)`, the same width in every post period) is gone.

## Reference and pins

Cases in `validation/r/cases/causal_synth_scpi.R` on `synth_qp_panel.csv` (T0 = 14, J = 8, T1 = 6):

- `scpi_quantities`: the weights, synthetic path, effects and pre-treatment fit, against the exact
  simplex QP (`quadprog` with a KKT polish).
- `scpi_default_intervals`: `scpi::scpi()` 3.0.1 (R 4.3.3, CVXR 1.0-15, ECOSolveR 0.6.2, Qtools 1.6.0,
  quantreg 6.1; scpi 4.x needs R >= 4.4) after `set.seed(8101)`, at its defaults, at
  `u.alpha = e.alpha = 0.025` and with `u.missp = FALSE`. It writes scpi's weights, rho, Sigma, the
  pseudo-residual mean, the 8 x 200 standard normal draws scpi takes (`scest()` consumes the generator
  first), the in-sample and per-method out-of-sample bounds and the intervals. scpi's ECOS solves of
  the in-sample programs mostly end at "Close to optimal solution found", about 1e-5 from the
  optimum, so the case also re-solves the same programs in `delta = x - w` with ECOSolveR at
  tolerance 1e-12 (`insample_tight_*`; those agree with p2a's solutions to about 1e-8 per draw).

The inference stage is compared on R's weights and R's draws (`scpi_inference` with the draws given),
so neither the weight solver nor the generator is part of the comparison.

| Test | Pins | Tolerance |
|---|---|---|
| `test_validate_scpi_simplex_weights` | weights (D01, D04, D05, D07 positive), synthetic path, effects, pre-RMSPE, pre-R2 | weights, effects 1e-6; statistics 1e-5 |
| `test_validate_scpi_default_intervals` | rho; Sigma; pseudo-residual mean; prediction; in-sample bounds (re-solved and scpi's); gaussian, ls and qreg bounds; the three intervals; run_scpi end to end on R's draws (scpi's ECOS weights and the gaussian interval) | rho 1e-12; Sigma, mean 1e-8; in-sample 1e-7 (re-solved), 1e-4 (scpi's); gaussian, ls 1e-6; qreg 1e-5; intervals 1e-4; end to end: weights 2e-5 absolute, interval 1e-3 |
| `test_validate_scpi_alpha_alias_and_no_misspecification` | at u.alpha = e.alpha = 0.025 (the `alpha = 0.05` alias): in-sample, gaussian, ls, qreg bounds, gaussian interval; with `u.missp = FALSE`: Sigma and in-sample bounds | as above |

Property tests (not registered): `test_scpi_interval_width_varies_over_post_periods` (before: width
1.5704 in every period), `test_scpi_interval_width_increases_with_level`,
`test_scpi_coverage_on_known_effect` (release only: 200 seeded DGPs with effect 2, period-level coverage
of the default interval at least 0.9 minus three Monte Carlo standard errors),
`test_scpi_seed_and_single_e_method`, `test_scpi_variance_method_alias`, `test_scpi_rejects_invalid_alpha`,
`test_scpi_rejects_too_few_sims`, `test_scpi_stops_when_cancelled`,
`test_scpi_penalised_intervals_are_ordered`, `test_qcqp_closed_forms`,
`test_e_method_parses_aliases_and_rejects_unknown`, `test_scpi_rejects_invalid_lambda`,
`test_scpi_rejects_non_finite_outcomes`, `test_variance_method_parses_aliases_and_rejects_unknown`,
`test_scpi_constraint_from_options`, and the lasso, ridge and lasso-simplex smoke tests.

## Timing (against v0.5.0)

Quiet re-time, 2026-11-15 (this tree at `2d0bb8b6`): release builds, `--no-default-features --features
ci`; one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; v0.5.0 (`f80b0131`) and this tree
interleaved per row with the order rotated per round, each process the median of its calls after a
warm-up (calls calibrated to about 0.8 s per process); cells are the median over five rounds (nine or ten
for rows within 1.05-1.15x), the ratio of the minima in brackets; 1-minute load average 1.0-3.0.
`run_scpi`, simplex, defaults (200 simulations), 10 post-treatment periods, on the deterministic noisy panel
described below; one call per process at J = 40:

| | J = 10, T0 = 20 | J = 40, T0 = 20 | J = 40, T0 = 50 |
|---|---|---|---|
| v0.5.0 closed-form heuristic interval | 28.6 µs | 36.7 µs | 289 µs |
| this tree, simulated intervals | 2.11 s | 2.6 s | 25.4 s |

Peak resident memory of the whole process (`/usr/bin/time -v`, one call, median of three) at J = 40, T0 = 50:
8.0 MB v0.5.0, 9.3 MB this tree.

**After CAUS-32 (scpi's intervals).** v0.5.0 computed the closed-form heuristic interval that
CAUS-32 replaced. The simulated intervals solve `2 x sims x T1` programs (4,000 at the defaults with
T1 = 10), so they cost orders of magnitude more; there is no like-for-like base. A first measurement
(2026-11-14, three processes of 1-3 calls) gave 1.98 s (J = 10, T0 = 20), 15.6 s (J = 40, T0 = 20)
and 19.7 s (J = 40, T0 = 50) per call. The table below is the final tree against the first version
of the simulated intervals (before the in-sample solves were certified centred and the retry with a
shorter step was added): release builds, `--no-default-features --features ci`, one core
(`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; five interleaved processes per side, each
the median of 5 calls (J = 10) or one call (J = 40); median of the five; 2026-11-14, load average
2-6. `run_scpi`, simplex, defaults (200 simulations, all three out-of-sample methods), T1 = 10, on
a deterministic noisy panel (donor k at period t: `10 + t (0.5 + 0.05 k) + sin(t (k + 1.3)) +
0.3 cos(7.1 t + 3.7 k)`; treated: 0.5, 0.3, 0.2 of donors 0-2 plus `0.2 sin(1.9 t)`); no
simulation failed on either side:

| | J = 10, T0 = 20 | J = 40, T0 = 20 | J = 40, T0 = 50 |
|---|---|---|---|
| final tree, seconds per call | 2.04 | 2.84 | 24.7 |
| first simulated version, seconds per call | 2.10 | 2.96 | 35.6 |
| ratio final / first | 0.97 | 0.96 | 0.69 |
| v0.5.0 closed form (not comparable), seconds per call | 1.83e-5 | not timed | not timed |

The first version returned the last barrier point even when its centring had stalled; on the
J = 40, T0 = 50 panel almost every solve stalled with the long step, so its bounds there were not
certified and the final tree's lower time comes with the certified ones. Peak resident memory
(`/usr/bin/time -v`, one call per process, J = 40, T1 = 10, 200 simulations): 4.6 MB for the final
tree at T0 = 20 and T0 = 50 (first version 4.7 MB); the simulation bounds are `2 x sims x T1`
numbers, and each program holds matrices of the donor count squared.

For scale, R's `scpi()` 3.0.1 on the case's panel (J = 8, T0 = 14, T1 = 6) takes about 5 seconds.
The time grows linearly with `sims`; the hosted limits cap `sims` and the call's work
(`scpi_work`: sims x T1 x max(J, 8)^2, the lasso weighted 8; a program's fixed cost does not
shrink with the donors, so a panel with fewer than 8 counts as 8). On release builds, one call per
process on one core, at the hosted value of 800,000 for J = 2-10 and over J = 10-160, T0 = 20-1,000,
T1 = 1-20 and every constraint (2026-11-15), a counted unit cost at most about 36 us, the ridge
constraint at J = 8-12 being the slowest: the slowest admitted call took about 29 s (simplex at most
about 16 s, lasso about 8 s). T0 is not counted. It changes the time of some panels several-fold
(the lasso at J = 40 and T1 = 5 took 3.9 s at T0 = 20 and 17.9 s at T0 = 50 for 20 simulations), but in
no measured call did it push the time per unit above that maximum. The simulated intervals' cost against v0.5.0's
closed form was signed off on 2026-11-14 (rule 5).

## Deviations

1. Weights: `run_scpi` solves the simplex QP exactly; scpi's `scest()` (CVXR, ECOS) stops about 1e-5
   from it, so the inference stage is pinned on R's weights and `run_scpi` end to end at 1e-3.
2. In-sample bounds: scpi's ECOS solves end about 1e-5 from the optimum ("Close to optimal solution
   found"); p2a matches the re-solved programs to 1e-7 and scpi's own bounds to 1e-4.
3. Random draws: `run_scpi` draws from a seeded `StdRng` (seed 42 by default), not R's generator, so
   its intervals for a given seed differ from scpi's by Monte Carlo error; the pins pass R's draws in.
4. Lasso and ridge intervals use the localised sets above and are not R-pinned.
5. scpi version: 3.0.1, the last release that installs on R 4.3.3; the defaults pinned here (rho
   type-2) are that version's.

## References

- Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). "Prediction Intervals for Synthetic Control
  Methods." *Journal of the American Statistical Association*, 116(536), 1865-1880.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). *JASA*, 105(490), 493-505.
- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). "Efficient Projections onto the
  l1-Ball for Learning in High Dimensions." *ICML 2008*.

## MCP extraction (T2-WP11)

The MCP tool `scpi` reads the unit column as a factor, the time column as a whole-number time
index and the outcome as a numeric column, in one request to the shared extractor with the
`Fail` policy. Units may be text or integer ids; they are the factor levels in R `factor()`
order (integer ids numerically, text in byte order), and the donors are the other units in that
order (before T2-WP11 it was hash order and changed from run to run). Times are the sorted
distinct values. With fewer rows than units x times cells the panel cannot be balanced, and the
tool error names the counts and the first empty cell (`scpi needs a balanced panel: 10 units x
20 times = 200 cells, but the data has 199 rows (unit 2 has no row for time 1940; cells
without a row: 1)`), found from the sorted cells of the rows. Otherwise the outcome grid is
filled in one pass: a second row for a (unit, time) cell (`unit 1 appears twice at time 1935`)
and a missing outcome are tool errors naming the unit and time or the column. Before, the
handler refused integer unit columns ("must be string type"; finding TOOL-5, scpi site).

Tests (`crates/p2a-mcp/src/tools/handlers/causal.rs`, on `validation/datasets/grunfeld.csv`;
not registered, no R value): `test_mcp_scpi_accepts_integer_unit_ids` and
`test_mcp_scpi_reports_unknown_units_and_missing_cells`.
