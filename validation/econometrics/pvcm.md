# Validation: Variable coefficients (pvcm) and mean group (pmg)

## Method Overview

`run_pvcm` and `run_pmg` (`crates/p2a-core/src/econometrics/panel/heterogeneous.rs`) fit one
OLS regression per entity, `y_it = X_it b_i + e_it`, and combine the entity estimates.

- **Within** (`PvcmType::Within`, R `pvcm(model = "within")`): the individual coefficients
  `b_i` and their SEs, `Var(b_i) = s_i^2 (X_i'X_i)^-1` with `s_i^2 = rss_i / (T_i - k)`. The
  reported overall coefficients are the unweighted mean of the `b_i`.
- **Random** (`PvcmType::Random`, Swamy 1970, R `pvcm(model = "random")`): with
  `Delta = (1/(N-1)) sum (b_i - b_bar)(b_i - b_bar)' - (1/N) sum Var(b_i)`, the GLS
  estimate `b = (sum W_i)^-1 sum W_i b_i`, `W_i = (Var(b_i) + Delta)^-1`, and its variance
  `(sum W_i)^-1`. When `D1 - D2` (the two terms of Delta) has a negative eigenvalue, Delta
  is `D1`, as plm, with a warning.
- **Mean group** (`run_pmg`, Pesaran and Smith 1995, R `pmg(model = "mg")`): the unweighted
  mean of the `b_i`, with the standard error of that mean,
  `sqrt(sum (b_i - b_bar)^2 / (N (N - 1)))` over the N entities fitted.
- **Homogeneity test** (both models, plm's `chisq.test`):
  `sum (b_i - b*)' Var(b_i)^-1 (b_i - b*)` with `b*` the estimate weighted by the
  `Var(b_i)^-1`, on `k (N - 1)` degrees of freedom.
- **Inference**: z statistics with normal p-values, as `summary.pvcm` and `summary.pmg`.
  `PvcmResult` reports the number of entities fitted (`n_entities_used`) next to all entity
  levels (`n_entities`), and the text shows "No. Entities: 12 (11 used)" when they differ.

Entities are fitted in R `factor()` order of the ids (numbers numerically, text in byte
order), so the Swamy sums are deterministic.

**Status**: R-pinned (methods `panel_pvcm_pmg` in `validation/methods/econ_panel_t3.json`
and `panel_pvcm_swamy_non_psd` in `validation/methods/econ_panel_t4.json`): the individual
coefficients and SEs, the Swamy coefficients, SEs, z, p-values and the diagonal of Delta
(positive-definite and D1 fallback), the homogeneity chi-square, and the mean-group
coefficients, SEs, z and p-values.

## Entities that cannot be estimated

An entity with no more rows than coefficients, or whose regressors are collinear, has no
`b_i`. It is skipped and named in `warnings`: "entity '12' skipped: 3 observations for 3
coefficients", "entity '3' skipped: its regressors are collinear (rank 2 of 3)". The text
output prints the warnings before the missing-value line, which stays last. A singular
`Var(b_i) + Delta`, a singular weight sum or a singular `Var(b_i)` in the homogeneity test is
an error naming the entity. Before Tier 3, short entities were dropped silently, collinear
entities were pseudo-inverted into the averages, and an entity with a singular
`Var(b_i) + Delta` was left out of the Swamy sums silently.

## R reference

Case `pvcm_pmg_sim` (`validation/r/cases/econ_panel_t3.R`, R 4.3.3, plm 2.6.7;
`validation/datasets/generated/pvcm_panel.csv`, N = 12 entities `id` 1..12, T = 20, n = 240,
heterogeneous slopes, positive-definite Swamy Delta):

```r
pd <- pdata.frame(d, index = c("id", "t"))
pvcm(y ~ x1 + x2, data = pd, model = "within")
pvcm(y ~ x1 + x2, data = pd, model = "random")
pmg(y ~ x1 + x2, data = pd, model = "mg")
```

| Quantity ((Intercept), x1, x2) | R |
|---|---|
| within, entity 1: coefficients | 1.191360514, -0.1501127754, -0.02595929598 |
| within, entity 1: SE | 0.3134300473, 0.1685327548, 0.1501410776 |
| random: coefficients | 0.6795444326, 0.5815089351, -0.4981040760 |
| random: SE | 0.4418901474, 0.1564522983, 0.08254980597 |
| random: Delta diagonal | 2.282425182, 0.2799281991, 0.06846336377 |
| random: homogeneity chi2 (df 33) | 3385.234629 |
| mg: coefficients | 0.6755447018, 0.5829283539, -0.4969613022 |
| mg: SE | 0.4420557664, 0.1565954001, 0.08313141715 |

The case file holds the coefficients and SEs of all twelve entities, and the full Delta
(off-diagonals -0.1129477848, -0.06503953957, 0.06155671950).

## Delta not positive semidefinite (D1 fallback)

Case `pvcm_swamy_grunfeld` (`validation/r/cases/econ_panel_t4.R`; Grunfeld, N = 10, T = 20,
`inv ~ value + capital`): the eigenvalues of `D1 - D2` are 0.0334, 0.00163 and -1120.48, so
plm uses `Delta = D1` (the case checks `r$Delta == D1`). p2a does the same and warns
"Swamy: Delta = D1 - D2 is not positive semidefinite (smallest eigenvalue -1.120478e3); D1 is
used, as plm". Before, p2a clamped the diagonal of `D1 - D2` at 0, which left the weights
indefinite and an intercept SE of 0.

| Quantity ((Intercept), value, capital) | R |
|---|---|
| random: coefficients | -9.629285137, 0.08458733660, 0.1994184033 |
| random: SE | 17.03503951, 0.01995590534, 0.05265335866 |
| random: Delta diagonal (D1) | 2344.244022, 0.003118178809, 0.02448242482 |
| homogeneity chi2 (df 27) | 901.4302336 |
| mg: coefficients | -21.36757126, 0.09128511040, 0.2052635409 |
| mg: SE | 15.31092428, 0.01765836575, 0.04947971788 |

## Tests

Registered (method `panel_pvcm_pmg`):

- `econometrics::panel::heterogeneous::tests::test_validate_pvcm_within_sim`: every
  entity's coefficients (`within_coef_<i>`) and SEs (`within_se_<i>`), and n;
- `econometrics::panel::heterogeneous::tests::test_validate_pvcm_random_sim`: the Swamy
  coefficients, SEs and Delta diagonal, and the homogeneity chi-square and its df;
- `econometrics::panel::heterogeneous::tests::test_validate_pmg_sim`: the mean-group
  coefficients and SEs, and n;
- `econometrics::panel::heterogeneous::tests::test_validate_pvcm_skips_collinear_entity_with_warning`:
  on the case data with entity 3's `x2` set to `2 * x1`, the within fit has 11 entities and
  the collinear warning, 12 entities of which 11 used, and the mean-group SE over the 11;
  with entity 12 cut to 3 rows, the short-entity warning (no R value compared).

Registered (method `panel_pvcm_swamy_non_psd`):

- `econometrics::panel::heterogeneous::tests::test_validate_pvcm_random_grunfeld_non_psd_delta`:
  the Swamy coefficients, SEs, z, p-values, Delta diagonal, homogeneity chi-square and df on
  Grunfeld, and the D1 warning;
- `econometrics::panel::heterogeneous::tests::test_validate_pmg_grunfeld`: the mean-group
  coefficients, SEs, z and p-values, and the homogeneity chi-square (it does not depend on
  Delta).

The tests of Tier 2's missing-value case `panel_grunfeld_na`
(`econometrics::panel::heterogeneous::tests::test_validate_pmg_grunfeld_na`,
`econometrics::panel::heterogeneous::tests::test_validate_na_pvcm_numeric_and_string_ids_agree`)
pin the mean-group coefficients and SEs on Grunfeld with missing cells and check that text and
numeric ids agree. With the numeric id the output's last line is then "Missing values: 5
of 200 rows dropped because of missing values (inv: 2, value: 1, firm_id: 2)".

The three full-Grunfeld PVCM and PMG tests, which compared the averages with rounded values
at tolerances up to 30, were deleted in Tier 3.

## Tolerances

`validation/README.md`, n = 240: coefficients 1e-8 relative, SE and statistics 1e-6
relative; n exact.

## Timing

Re-timed on a quiet machine on 2026-11-13: release builds of v0.5.0 (`f80b0131`) and this version,
one P core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), five rounds interleaving the two builds in
alternating order; each round is one process per build reporting the median of its calls after
warm-up calls. The table gives the ratio of the medians of the five per-round medians, with the
ratio of their minima in brackets (1-minute load average 1.7-2.6). Simulated panels of n / 20
entities over 20 periods with two regressors.

| Fit | n | v0.5.0 µs | new µs | new / v0.5.0 [min] |
|---|---|---|---|---|
| within (one OLS per entity) | 100 | 17.6 | 19.5 | **1.10** [1.08] |
| within (one OLS per entity) | 1,000 | 162 | 177 | 1.09 [1.09] |
| within (one OLS per entity) | 10,000 | 1,496 | 1,579 | 1.06 [1.07] |
| random (Swamy) | 100 | 24.3 | 27.3 | **1.12** [1.15] |
| random (Swamy) | 1,000 | 216 | 232 | 1.07 [1.07] |
| random (Swamy) | 10,000 | 1,925 | 1,975 | 1.03 [1.04] |

Both fits stay within 1.1x of v0.5.0 from n = 1,000 on; at n = 100 the within fit is at 1.10x
(17.6 µs against 19.5 µs) and the Swamy fit at 1.12x (24.3 µs against 27.3 µs; 1.15x by the
minima), which is above the rule and not signed off. The added cost is the per-entity
bookkeeping (`entities`, `n_entities_used`), the eigen check of Delta in the Swamy fit and the
weighted homogeneity statistic, a constant of about 2-3 µs; none of it grows faster than the
number of entities.

## Known differences

- The CLI and MCP tables still print `t` headers over the z statistics (to be relabelled
  there).

## References

- Swamy, P. A. V. B. (1970). Efficient inference in a random coefficient regression model.
  *Econometrica*, 38(2), 311-323.
- Pesaran, M. H., & Smith, R. (1995). Estimating long-run relationships from dynamic
  heterogeneous panels. *Journal of Econometrics*, 68(1), 79-113.
- Croissant, Y., & Millo, G. (2008). Panel data econometrics in R: the plm package. *Journal
  of Statistical Software*, 27(2).
