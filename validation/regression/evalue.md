# E-values (`regression::evalue`)

## Method

VanderWeele & Ding (2017): `E = RR + sqrt(RR (RR - 1))` for `RR >= 1` (the reciprocal below 1), at
the point estimate and at the confidence limit closer to 1 (1 when the interval contains 1). Odds
ratios: RR = OR for a rare outcome, `sqrt(OR)` otherwise. Hazard ratios: RR = HR for a rare outcome,
`(1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR))` otherwise (VanderWeele 2020, Biometrics 76(3)).
Standardised mean differences: `RR = exp(0.91 d)` (Chinn 2000).

## R reference

`EValue` 4.1.4, case `evalue_formulas` (inputs in `evalue_inputs.csv`): `evalues.RR`, `evalues.OR`,
`evalues.HR`, `evalues.MD`.

## Values

| row | call | RR | E (point) | E (CI) |
|---|---|---|---|---|
| 1 | RR 2.5 (1.8, 3.5) | 2.5 | 4.436491673103708 | 3 |
| 2 | RR 0.6 (0.45, 0.8) | 0.6 | 2.7207592200561264 | 1.8090169943749475 |
| 3 | RR 2 (0.9, 3) | 2 | 3.414213562373095 | 1 |
| 4 | OR 2.5 (1.5, 4), rare | 2.5 | 4.436491673103708 | 2.3660254037844384 |
| 5 | OR 4 (2, 8), common | 2 | 3.414213562373095 | 2.179580427103275 |
| 6 | HR 1.5 (1.2, 1.9), rare | 1.5 | 2.3660254037844384 | 1.6898979485566354 |
| 7 | HR 4, common | 2.560660171779822 | 4.559740046011876 | 2.606409265530888 |
| 8 | MD 0.5, se 0.1 | 1.5761733830339912 | 2.5291419818612693 | 1.9680373469882557 |

`evalue_rr(3.9)` = 7.263034344160047.

## Tier 4: hazard ratios with a common outcome (`evalue_hr_common`)

Registry entry `evalue_hr_common` (`validation/methods/regression_sensitivity_t4.json`). R:
`evalues.HR(0.6, 0.4, 0.9, rare = FALSE)`, `evalues.HR(1.3, 0.9, 1.8, rare = FALSE)`,
`evalues.HR(0.5, NA, 0.8, rare = FALSE)` (inputs `evalue_hr_inputs.csv`).

| row | HR (CI) | RR | E (point) | E (CI) |
|---|---|---|---|---|
| hr1 | 0.6 (0.4, 0.9) | 0.7025644572836847 | 2.199622395659878 | 1.3612206313389328 |
| hr2 | 1.3 (0.9, 1.8) | 1.1992701292249968 | 1.6881246204862959 | 1 |
| hr3 | 0.5 (-, 0.8) | 0.620136837873846 | 2.6064092655308877 | 1.6088913483778375 |

`test_validate_evalue_hr_common` pins RR, both E-values and the RR-scale limits (through the
conversion; `EValueResult` carries no limits), `Tol::new(1e-12, 1e-14)`.

## Tolerances

Closed forms: `Tol::new(1e-12, 1e-14)`.

## Tests

`test_validate_evalue_formulas` (registry entry `evalue`). The old `test_validate_against_r`
(tolerances 0.01-0.05 on rounded values) is deleted.

## Deviations

None for the pinned rows. Row 7 (HR, common outcome) uses EValue's conversion and is pinned with its
RR-scale limits; `row8_e_ci` uses EValue's `exp(0.91 d - 1.78 se)` and is pinned.
`could_explain_away(observed_rr, rr_eu, rr_ud)` and `bias_factor` have no R function; they are
checked as properties (`test_could_explain_away_at_the_evalue`): a confounder with both associations
at the E-value explains the observed RR away, B(E, E) = E^2 / (2E - 1) = RR (RR 3.9 and 0.6), B(8, 8) =
4.27 explains RR 3.9 away while B(3, 3) = 1.8 does not, and `bias_factor(evalue_rr(x)) = x` within
1e-12 for x = 1.2, 2.5, 3.9.

## Wall-clock against v0.5.0 (2026-11-13)

Re-timed on 2026-11-13: release builds of v0.5.0 (`f80b0131`) and this version, one P core
(`taskset -c 2`, `RAYON_NUM_THREADS=1`), five rounds interleaving the two builds in alternating order;
each round is one process per build reporting the median of its calls after warm-up calls. Ratios are
of the medians of the five per-round medians, with the ratio of their minima in brackets (1-minute
load average 4.3-8.7 at the start of each round, right after the conversation-store benchmarks of the
round before, and 2.2-3.8 at its end; the alternating order protects the ratios).
`evalue_hr(hr, lo, hi, rare = false)`, one call per hazard ratio over a grid of n ratios from 1.05 to
3.05, limits at 0.8 and 1.25 times the ratio:

| calls | v0.5.0 µs | new µs | new / v0.5.0 [min] |
|---|---|---|---|
| 100 | 0.62 | 3.53 | **5.70** [5.48] |
| 1,000 | 5.97 | 35.6 | **5.97** [6.32] |
| 10,000 | 62.2 | 361 | **5.80** [5.78] |

About 36 ns a call against 6 ns: the common-outcome conversion is now VanderWeele's
`(1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR))` (EValue's), where v0.5.0's was approximate. The unified
`E_Value` risk-ratio row is 0.93-1.00x. The rows above 1.1x were signed off on 2026-11-13 (rule 5).
