# Matching (MatchIt and optmatch) Validation

p2a's matching (`p2a_core::econometrics::matching`) computes what MatchIt 4.8.0 computes for the ATT, with
optimal full matching solved as `optmatch::fullmatch` (optmatch 0.10.8) does. The R reference cases are in
`validation/r/cases/causal_matching.R`; the registry is `validation/methods/causal_matching.json`.

## Methods and R equivalents

| p2a | R |
|-----|---|
| `MatchMethod::NearestNeighbor { ratio, caliper, replace }`, `match_it_with(.., &MatchOptions { m_order, seed })` | `matchit(method = "nearest", ratio, caliper, replace, m.order)` |
| `MatchMethod::CoarsenedExact { cutpoints: None, n_bins: None }` | `matchit(method = "cem")` (`cutpoints = "sturges"`, the default) |
| `MatchMethod::CoarsenedExact { n_bins: Some(k), .. }` | `matchit(method = "cem", cutpoints = "q<k>")` |
| `MatchMethod::CoarsenedExact { cutpoints: Some(c), .. }` | `matchit(method = "cem", cutpoints = list(..))` |
| `MatchMethod::Full { min_ratio, max_ratio }` | `matchit(method = "full", min.controls, max.controls)` -> `optmatch::fullmatch` |
| `MatchMethod::Subclass { n_subclasses }` | `matchit(method = "subclass", subclass = k, min.n = 0)` |

Distances: `DistanceMethod::Logit` and `Probit` are `distance = "glm"` with `link = "logit"` or `"probit"` (the
propensity score on the probability scale); `Mahalanobis` and `Euclidean` are `distance = "mahalanobis"` and
`"euclidean"`. CEM uses no distance. `MOrder` is `m.order` (`largest`, `smallest`, `data`, `random`); the default
is `largest` for a propensity score and `data` for a covariate distance, and matching with replacement always runs
in data order, as MatchIt. Only the ATT is estimated.

## Algorithms

- **Propensity score.** `econometrics::glm::fit_propensity` with the requested link, an intercept and the
  covariates: R's `glm(binomial(link))`. After glm's deviance rule the coefficients are refined until their
  relative change is below `PROPENSITY_COEF_TOL` (1e-10), so the scores are the MLE to about 1e-10 (R at
  `epsilon = 1e-14`). Separation and non-convergence are reported in `warnings` and `propensity_convergence`.
- **Nearest neighbour** (`matchit2nearest`, `nn_matchC_vec`). Treated units choose in `m.order`; `ratio = k` is
  matched in k rounds (every treated unit gets its first control before any gets a second). The caliper is
  `caliper * sd(ps)` with R's `sd`. Without replacement a used control is removed through skip lists; an exact
  tie in the scalar distance goes to the control below, as `nn_matchC`. Weights are match-matrix weights
  (`get_weights_from_mm`) scaled by `.make_sum_to_n`, so the positive weights of each group sum to its count.
- **Covariate distances** (`transform_covariates`, `pooled_cov`). Constant columns are dropped. The Mahalanobis
  metric uses the pooled within-group covariance; its inverse is `linalg::solve::sym_inverse` with the
  generalised policy (a warning when singular), and the covariates are whitened once (`z = x Q diag(sqrt(l))`), so
  every distance is a Euclidean distance of whitened rows. With one covariate MatchIt switches to the scalar rule
  and so does p2a. Above the parallel threshold the search per treated unit runs on rayon.
- **CEM** (`cem_matchit`). Each covariate is cut at its breaks: Sturges (`nclass.Sturges`, `ceiling(log2(n) + 1)`
  equal-width bins over the range), `"q<k>"` (type-7 quantiles at `1/k, ..., (k-1)/k`) or the given cut points,
  each padded with `-Inf` and `Inf`; intervals are left-closed, as `findInterval`. Strata holding both groups are
  kept and numbered in order of first appearance. Weights are `get_weights_from_subclass` (treated 1, controls
  `n_t(s) / n_c(s)`), normalised per group.
- **Subclassification** (`matchit2subclass`). The cut points are the type-7 quantiles of the treated propensity
  scores at `seq(0, 1, length.out = k + 1)`; units are assigned with `findInterval(.., all.inside = TRUE)`. A
  subclass without treated or control units is discarded with a warning (`min.n = 0`). Weights as CEM.
- **Optimal full matching** (`matchit2full`, `fullmatch`, `fmatch`). The distance matrix is treated x controls:
  `|ps_t - ps_c|` on the probability scale, or the covariate metric above. optmatch's network (Hansen and Klopfer
  2006): every treated unit supplies `max.controls`, one unit of flow per treated-control arc, overflow arcs to an
  end node and control-to-sink arcs, with optmatch's mapping of `min.controls` and `max.controls`. When
  `max_ratio` cannot absorb every control, optmatch's recovery solves the unconstrained problem, takes the omit
  fraction from the controls it could use and solves again leaving that many controls unmatched (a warning says
  how many); `max_ratio <= 0.5` is solved transposed. Costs are integers at resolution `max(d) / 2^40`, so the
  solution is the exact optimum (optmatch's default `tol = 0.001` stops within `tol * n` of it). Sets are the
  connected components of the flow and are numbered by their smallest row; weights as CEM.
- **Network simplex** (`econometrics::min_cost_flow`, LEMON's `NetworkSimplex`). An artificial root and big-M
  arcs give the first spanning tree; entering arcs come from block search (block size `max(ceil(sqrt(m)), 10)`),
  the leaving arc by Cunningham's rule (strongly feasible trees, no cycling); potentials are `i128`. Costs are
  bounded by 2^52 and capacities by 2^62; a pivot guard and the budget checkpoint bound the loop.
- **Balance** (`summary.matchit(standardize = TRUE)`, `bal1var`, `qqsum`). Weighted means; the standardised
  mean difference over the treated standard deviation of the full sample (MatchIt's `pooled_sd` fallback when it
  is 0); the weighted variance ratio (`col_w_var`; NA for binary covariates); the weighted eCDF mean and max, the
  running difference of the normalised group weights taken at the last unit of each run of equal values (with
  unit weights the max is the two-sample KS statistic). `ks_statistic` equals `ecdf_max`. The ESS per group is
  Kish's `(sum w)^2 / sum w^2`; `effective_sample_size` is their sum.

## Limits

Optimal full matching needs the dense treated x control problem (about 48 bytes per pair). As optmatch by
default, problems above `FULL_MATCH_MAX_PAIRS` = 1e7 treated-control pairs are refused in every mode, before the
distance matrix is built. Nearest-neighbour lists, stratum control lists and the network are allocated fallibly
under the allocation ceiling, and every user-sized loop checks the budget.

## Reference

- Cases: `matchit_nearest`, `matchit_full_cem`, `matchit_subclass`, `mcf_networks` (`validation/expected/`), from
  `validation/r/cases/causal_matching.R`.
- Datasets: `validation/datasets/generated/matching_sim.csv` (seed 20261128; n = 240, 89 treated, covariates
  `x1`, `x2`, binary `x3`) and `validation/datasets/generated/mcf_networks.csv` (five seeded networks, one
  infeasible).
- Versions: R 4.3.3, MatchIt 4.8.0, optmatch 0.10.8, rlemon 0.2.1 (LEMON 1.3), lpSolve 5.6.23.

```bash
Rscript validation/r/run.R --check matchit_nearest matchit_full_cem matchit_subclass mcf_networks
```

## Values

| Quantity | R value |
|---|---|
| logit glm coefficients | (Intercept) -0.6541816134792587, x1 0.6850412954942849, x2 -0.342458183526464, x3 0.24926187375930156 |
| probit glm coefficients (epsilon 1e-14) | -0.4041817140727972, 0.4135301571456666, -0.20590999220505532, 0.1591754938249637 |
| first pairs (treated rows 5, 6, 9, 13, 20) | largest 25, 137, 231, 227, 55; data 25, 222, 231, 185, 151; smallest 25, 236, 231, 75, 83; Mahalanobis 95, 104, 221, 123, 184 |
| 1:2 matching | 62 treated with 2 matches, 27 with 1; weights 0.848314606741573, 1, 1.696629213483146; control ESS 136.5689655172414 |
| caliper 0.1 | 0.015051860900814057 on the score scale; 74 treated matched |
| SMD before (x1, x2, x3) | 0.5380904131926535, -0.2495286642141948, 0.118024459502027 |
| SMD after, 1:1 largest | 0.17246641030764848, -0.025397154512881485, 0 |
| full, PS, unconstrained | optimum 0.8659137461118286 (MatchIt's default tol: 0.8700895944690001) |
| full, PS, [0.5, 2] | optimum 9.14686351578491 (default 9.147925877997729) |
| full, probit PS, unconstrained | 0.9093516377499569 |
| full, Mahalanobis | unconstrained 65.97481807569326 (79 sets); [0.5, 2] 97.66738384914133; max 1 29.943537665457278 (72 controls unmatched); max 0.5 (flipped, 129 rows) 71.889278613498 |
| CEM | Sturges: 68 treated / 90 controls kept, 33 strata; `"q4"` 30 strata; explicit cut points 22 strata |
| subclass cut points, logit | 0.13302953290096708, 0.3024346212014255, 0.381374629249142, 0.4686995918167506, 0.562459234757879, 0.8361914611686652 |
| network simplex costs | 340, 244, 213, 151, infeasible |

Full precision is in the case JSON.

## Tests and tolerances

R-pinned (registry `validation/methods/causal_matching.json`):
- `test_validate_matchit_propensity_models`: logit and probit coefficients and scores against the epsilon 1e-14
  refit (`Tol::iterative(1e-8)`) and MatchIt's own glm (logit 1e-8, probit 1e-5: MatchIt's glm stops 4.6e-7 away).
- `test_validate_matchit_probit_in_every_ps_method`: the probit score reaches nearest, full and subclass matching.
- `test_validate_matchit_nearest_logit_largest`, `test_validate_matchit_nearest_logit_data`,
  `test_validate_matchit_nearest_probit`, `test_validate_matchit_nearest_mahalanobis`,
  `test_validate_matchit_nearest_euclidean`: pairs exact.
- `test_validate_matchit_nearest_ratio2_rounds`: the 1:2 match matrix exact, weights 1e-8, ESS 1e-6.
- `test_validate_matchit_nearest_caliper`: the absolute caliper (`Tol::iterative(1e-8)`), pairs, weights.
- `test_validate_matchit_nearest_replace`: weights with replacement.
- `test_validate_matchit_balance_table`: balance before and after matching (means, SMD, variance ratio, eCDF mean
  and max) at 1e-6 (`Tol::stat`).
- `test_validate_matchit_m_order_random_is_seeded`, `test_validate_matchit_display_shows_the_balance_tables_and_notes`,
  `test_validate_matching_rejects_invalid_options`: on the `matchit_nearest` data; the random order is p2a's
  seeded shuffle, not R's.
- `test_validate_full_matching_total_distance`, `test_validate_full_matching_bounded`,
  `test_validate_full_matching_probit`: PS totals at `Tol::iterative(1e-8)`, below MatchIt's default solution;
  every unit kept, stars within the bounds, the weight formula on p2a's partition.
- `test_validate_full_matching_weights`: MatchIt's weights from R's own partitions.
- `test_validate_full_matching_mahalanobis`, `test_validate_full_matching_recovery`,
  `test_validate_full_matching_flipped`: partitions exact, weights and totals 1e-8.
- `test_validate_full_matching_rejects_infeasible_and_invalid_bounds`: infeasible and invalid bounds refused.
- `test_validate_cem_sturges`, `test_validate_cem_quantile_and_explicit_cut_points`: weights 1e-8, counts exact.
- `test_validate_subclass_logit_probit`: cut points `Tol::iterative(1e-8)`, labels exact, weights 1e-8.
- `test_validate_mcf_networks` (`econometrics::min_cost_flow`): costs exact, optimality conditions checked.

Not R-pinned:
- `test_validate_ks_before_matching_vs_matchit` (T0-WP11): the before-matching KS statistic (eCDF max) of x1 and
  x2 of `create_matchit_validation_dataset()` equals MatchIt 4.8.0's 0.34643081041698437 and 0.16215117165658516
  within 1e-12; values typed into the test, a quoted-value check.
- `test_ecdf_max_is_the_ks_statistic_with_unit_weights` (ported from T0-WP11): two `ks.test(x, y)$statistic`
  values from R typed into the test, compared with `ecdf_stats` (unit weights) within 1e-12.
- `test_find_interval_is_left_closed`, `test_cem_without_common_strata_is_an_error`,
  `test_subclass_without_controls_is_discarded_with_a_warning`,
  `test_full_matching_refuses_problems_above_the_pair_cap`, and the network simplex property tests.

## Deviations from MatchIt and optmatch

- Full matching is solved exactly; MatchIt's default `tol = 0.001` is 0.48% above the optimum on `matching_sim`.
  The pins are optmatch at `tol = 1e-8`.
- On a propensity-score distance the optimal full matching is not unique (several stars cost the same on a line):
  totals, structure and the weight formula are pinned, partitions only on the Mahalanobis distance.
- The full-matching distance is on the probability scale (`distance = "glm"`, `link = "logit"`), as MatchIt;
  MatchIt's logit scale needs `link = "linear.logit"`, which p2a does not offer.
- Subclassification uses `min.n = 0`: MatchIt's default `min.n = 1` moves units between adjacent subclasses with
  the C++ routine `subclass_scootC`, which is not ported. Identical whenever every subclass holds both groups.
- `m.order = "closest"` and `"farthest"` are refused; the random order is a seeded ChaCha8 shuffle (seed
  `MatchOptions::seed`, default 42), not R's RNG.
- Exact ties among covariate distances with two or more covariates go to the lowest row (MatchIt's
  `nn_matchC_mahcovs` order is undocumented).
- Calipers must be positive and finite; on Mahalanobis and Euclidean distances the caliper is p2a's extension in
  raw distance units (MatchIt requires covariate-named calipers there).
- CEM `n_bins: Some(1)` is one bin and `Some(2)` cuts at the median (MatchIt errors on `"q1"` and reads the one
  `"q2"` quantile as a bin count).
- `ks_statistic` is summary.matchit's weighted eCDF max; before matching it is the two-sample KS statistic.
- The treatment must be coded 0/1.
- `effective_sample_size` is `ess_treated + ess_control`, MatchIt's per-group Kish ESS.

## Not covered

Estimands ATE and ATC; `exact`, `mahvars`, `antiexact`, `discard`, `reestimate`, `s.weights`, `k2k`, `grouping`;
the methods genetic, optimal pair, cardinality and quick; BART, GBM and GAM distances; variable-ratio nearest
neighbour (`min.controls`/`max.controls` of nearest); `std.caliper = FALSE`; covariate-named calipers; MatchIt's
`"scott"` and `"fd"` CEM rules; full-matching calipers.

## Legacy

`validation/expected/match_nn.csv` (written by `validation/scripts/validate_causal_design.R`, never asserted) is
superseded by the `matchit_nearest` case.

## Missing values and column types

Missing values (null or NaN) in the treatment or covariates are refused, with "Column 'x' contains k missing
value(s) (null or NaN). Drop or impute them first." (`EconError::NullValues`, naming the column and the count). A
text column gives `NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are read as numbers and give
the same result as Float64. Results on complete data are unchanged.

R: MatchIt 4.8.0 refuses them: "Missing and non-finite values are not allowed in the covariates." and "Missing
values are not allowed in the treatment." (checked for T2-WP13).

Tests: `test_validate_na_match_it_columns` (case `causal_mediation_na`, registry `validation/methods/causal_na.json`,
id `weighting_na_refusal`).

## References

- Ho, D.E., Imai, K., King, G., & Stuart, E.A. (2007). Matching as Nonparametric Preprocessing for Reducing Model
  Dependence in Parametric Causal Inference. *Political Analysis*, 15(3), 199-236.
  https://doi.org/10.1093/pan/mpl013
- Ho, D.E., Imai, K., King, G., & Stuart, E.A. (2011). MatchIt: Nonparametric Preprocessing for Parametric Causal
  Inference. *Journal of Statistical Software*, 42(8), 1-28. https://cran.r-project.org/package=MatchIt
- Iacus, S.M., King, G., & Porro, G. (2012). Causal Inference without Balance Checking: Coarsened Exact Matching.
  *Political Analysis*, 20(1), 1-24. https://doi.org/10.1093/pan/mpr013
- Hansen, B.B. (2004). Full Matching in an Observational Study of Coaching for the SAT. *Journal of the American
  Statistical Association*, 99(467), 609-618. https://doi.org/10.1198/016214504000000647
- Hansen, B.B., & Klopfer, S.O. (2006). Optimal Full Matching and Related Designs via Network Flows. *Journal of
  Computational and Graphical Statistics*, 15(3), 609-627. https://doi.org/10.1198/106186006X137047
- Kovács, P. (2015). Minimum-cost flow algorithms: an experimental evaluation. *Optimization Methods and
  Software*, 30(1), 94-127. https://doi.org/10.1080/10556788.2014.895828
