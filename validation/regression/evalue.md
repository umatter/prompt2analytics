# E-values (`regression::evalue`)

## Method

VanderWeele & Ding (2017): `E = RR + sqrt(RR (RR - 1))` for `RR >= 1` (the reciprocal below 1), at
the point estimate and at the confidence limit closer to 1 (1 when the interval contains 1). Odds
ratios: RR = OR for a rare outcome, `sqrt(OR)` otherwise. Hazard ratios: RR = HR for a rare outcome.
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

## Tolerances

Closed forms: `Tol::new(1e-12, 1e-14)`.

## Tests

`test_validate_evalue_formulas` (registry entry `evalue`). The old `test_validate_against_r`
(tolerances 0.01-0.05 on rounded values) is deleted.

## Deviations

- REG-5: row 7 (HR, common outcome): EValue converts by `(1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR))`;
  p2a uses `sqrt(HR)`.
- Not pinned (new observation): `row8_e_ci`. EValue uses `exp(0.91 d - 1.78 se)` (1.9680373469882557),
  p2a `exp(0.91 (d - 1.96 se))` (1.9669630004308964).
- REG-39: `could_explain_away` is not pinned.
