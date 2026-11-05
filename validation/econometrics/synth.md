# Validation: Synthetic Control Method (SCM)

## What p2a computes

`run_synthetic_control` (`crates/p2a-core/src/econometrics/synth.rs`) follows Abadie, Diamond and
Hainmueller (2010) and R's `Synth::synth()`:

- **Predictors** (`k`, aggregated over their time windows) are divided by their standard deviation
  across the treated unit and the donors (n - 1 denominator), as `synth()` does; a predictor whose
  donor values do not vary is refused ("no variation"). V, the predictor loss and the reported
  predictor weights are on this standardised scale; the predictor balance is on the raw scale.
- **Unit weights for a given V** are the exact solution of the simplex QP
  `min (X1 - X0 w)' V (X1 - X0 w)`, `w >= 0`, `sum w = 1`: a primal active-set method on the problem
  in factor form, normalised by `trace(H) / J`, with a tie-breaking ridge
  `SIMPLEX_QP_RIDGE = 1e-10` (relative). When several weight vectors fit equally well (a face of
  optima, typical with more donors than predictors) the ridge picks the minimum-norm one and a
  warning says the weights are not unique. Bitwise invariant to a common power-of-2 scale.
- **V**: `Equal` (1/k), `Custom` (finite, non-negative, positive sum; normalised to sum 1) or
  `DataDriven`: Nelder-Mead on the pre-treatment MSPE from two starts, equal V and Synth's
  regression-based V (`SV2`: squared slopes of the pre-period outcomes on the standardised
  predictors), keeping the lower loss (equal V on ties), as `synth()` does; R `optim`'s `reltol`
  stopping rule; `convergence` and `v_start_losses` are reported.

## Reference and pins

Case file `validation/r/cases/causal_synth_scpi.R`, case `synth_weights_qp`, on
`validation/datasets/generated/synth_qp_panel.csv` (treated `TR`, donors `D01`..`D08`, 20 periods,
treatment at 15, predictors x1..x3 and x1_big..x3_big = 1e4 x). The QP is solved by
`quadprog::solve.QP` with a KKT polish on the standardised predictors; the standardisation and SV2
are written out and checked against Synth 1.1.10's source expression. `Synth::synth()` itself is
not called (its V search uses `optimx` and `ipop`).

| Test | Pins | Tolerance |
|---|---|---|
| `test_validate_synth_weights_qp` | equal V: weights, predictor loss, pre-MSPE, effects, average effect, synthetic path, synthetic predictor values (raw scale), number of positive weights | weights, effects 1e-6; loss, MSPE 1e-5 |
| `test_validate_synth_weights_qp_custom` | V = (0.6, 0.3, 0.1): weights, loss, MSPE, average effect | as above |
| `test_validate_synth_weights_qp_invariant_to_predictor_units` | x 1e4 predictors (all, and x2 alone) give the same weights | as above; 1e-10 between fits |
| `test_validate_synth_regression_start` | SV2 start | 1e-6 |

Measured agreement is about 1e-13. `SIMPLEX_QP_RIDGE` is part of the pinned definition: the exact
KKT solution without it differs from `equal_w` by 4.8e-6 relative (D05).

Property tests (not registered): `test_simplex_qp_exact_cases`, `test_synth_invalid_v_is_error`,
`test_synth_predictors_without_variation_is_error`, `test_synth_duplicate_donors_split_and_warn`,
`test_synth_datadriven_keeps_the_better_start`,
`test_synth_datadriven_is_unit_invariant_and_not_worse_than_equal`,
`test_synth_regression_start_falls_back_to_equal_v`, `test_synth_many_donors_inside_hull`, and the
known-DGP tests `test_synth_weight_recovery_recovers_dgp`, `test_synth_treatment_effect_recovers_dgp`,
`test_synth_time_series_alignment_recovers_dgp`.

## Deviations

1. The data-driven V search has no R pin (no reference for Synth's `optimx` path): property tests
   only.
2. On a face of optima the minimum-norm weights of the ridge are reported with a warning; Synth's
   `ipop` returns some interior point of the face.

## Not run (Synth not installed in the validation environment)

The California tobacco example (`Synth::synth` on `smoking`) and its placebo inference are not part
of the pins; p2a's placebo ranks are tested by `test_synth_placebo_inference_ranks_every_unit`.

## Missing values and column types

Data requirements of `run_synthetic_control` (MCP `synthetic_control`, CLI `causal synth`):

- one row per (unit, time): a duplicate row is "synthetic control needs one row per (unit,
  time): {k} duplicate row(s) (first: unit '{u}', time {t}). Remove the duplicates first.";
- unit and time labels are never missing: a null label is `NullValues` naming the column;
- the outcome is present in every optimisation-window and post-treatment cell of the treated
  unit and the donors: a missing value there is "Outcome '{outcome}' is missing (null or NaN)
  for unit '{u}' at time {t}.", an absent cell "No data for unit '{u}' at time {t}";
- predictors are aggregated (mean, first, last or sum) over the values present in their
  window;
- an infinite value (`Inf` or `-Inf`) anywhere in the outcome or a predictor column is refused
  before the fit, whichever unit or time it belongs to: "Column '{column}' contains {k}
  infinite value(s); remove or recode them before estimating.";
- integer outcomes and predictors are read as numbers; unit ids may be text, integers or
  floats (`treated_unit` is matched against the label, e.g. "3"); times are integers,
  whole-number floats or dates.

Donors are in sorted order (R `factor()` order); they followed a hash-set order. The panel is
indexed once, so building the matrices is O(n) instead of one frame scan per cell. Weights
and effects on valid data are unchanged.

Tests (p2a-core, `econometrics::synth::tests`): `test_na_synth_integer_outcome_and_unit_ids`,
`test_na_synth_duplicate_cell`, `test_na_synth_missing_outcome_cell`,
`test_na_synth_null_unit_label`. Not R-pinned (Synth is not installed here).

## References

- Abadie, A. & Gardeazabal, J. (2003). "The Economic Costs of Conflict: A Case Study of the Basque Country." *American Economic Review*, 93(1), 112-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies." *JASA*, 105(490), 493-505.
- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391-425.
