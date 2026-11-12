# Validation: Exact Binomial Test (binom.test)

## Method Overview

`binom_test(x, n, p, alternative, conf_level)` tests H0: the success probability is `p` from `x`
successes in `n` trials. One-sided p-values are binomial tails, the two-sided p-value sums the
probabilities of the outcomes no more likely than the one observed, and the confidence interval
is Clopper-Pearson (Beta quantiles).

## Reference

| Package | Language | Function | Version | Case |
|---------|----------|----------|---------|------|
| stats | R | `binom.test` | R 4.3.3 | `stat_rank_binom` |

## Method

- **One-sided p-values:** `less` is P(X <= x) (the binomial cdf), `greater` is P(X >= x), computed
  as the survival function P(X > x - 1) (`statrs` `sf`), not as 1 - cdf.
- **Two-sided p-value (R's rule):** the sum of P(X = k) over the outcomes k with
  P(X = k) <= P(X = x) (1 + 1e-7), and 1 when x = np. The other tail is the suffix of
  [ceil(np), n] or the prefix of [0, floor(np)] whose cells qualify, found by binary search (the
  pmf is monotone on each side of the mean); it adds nothing when no cell qualifies.
- **Confidence interval:** Clopper-Pearson through the Beta quantile, two-sided or one-sided as
  the alternative asks; the estimate is x / n.

## Pinned values

Case `stat_rank_binom` (`validation/r/cases/stats_rank_t3.R`; dataset
`validation/datasets/generated/rank_binom_grid.csv`, one row per cell, column `stat15` marking the
cells of the former STAT-15 difference), registry `validation/methods/stats_rank_t3.json`, id `binom_test`. Tolerances: the
`validation/README.md` rows at the smallest n of the grid (10): p `Tol::p_value` (1e-5 relative,
1e-12 absolute), interval bounds 1e-5 relative, the estimate `Tol::coef`; the tiny upper tails
below at 1e-10 relative.

Every value is pinned in all 23 cells: the two-sided p (under `p_two`, and under `p_two_stat15`
for cells r04-r12), the one-sided p-values, the two-sided interval, the one-sided bounds
(`ci_greater_lower`, `ci_less_upper` in the case) and the estimate.

| Cell | (x, n, p) | R two-sided p | R p (less) | R p (greater) | R 95% CI |
|---|---|---|---|---|---|
| r01 | (15, 100, 0.1) | 0.09628404792835761 | 0.9601094728889166 | 0.07257296526488087 | [0.08645438564164856, 0.23530750014895485] |
| r02 | (0, 10, 0.3) | 0.038839603299999984 | 0.028247524899999987 | 1.0 | [0.0, 0.30849710781876083] |
| r03 | (10, 20, 0.5) | 1.0 | 0.5880985260009763 | 0.5880985260009763 | [0.27195784956079183, 0.7280421504392082] |
| r04 | (3, 10, 0.37) | 0.753923870210605 | 0.45999620731185487 | 0.7794292162670167 | [0.0667395111777345, 0.6524528500599973] |
| r05 | (7, 10, 0.75) | 0.7184324264526372 | 0.4744071960449222 | 0.7758750915527343 | [0.3475471499400027, 0.9332604888222655] |
| r06 | (1, 10, 0.05) | 0.40126306076162127 | 0.9138616441006835 | 0.40126306076162127 | [0.0025285785444617865, 0.44501611702819543] |
| r07 | (2, 10, 0.05) | 0.0861383558993165 | 0.9884964426207031 | 0.0861383558993165 | [0.025210726326833365, 0.5560954623076414] |
| r08 | (5, 100, 0.01) | 0.0034323215877545264 | 0.9994654655360069 | 0.0034323215877545264 | [0.016431879182052158, 0.11283491110546273] |
| r09 | (6, 50, 0.05) | 0.03777617298977736 | 0.9882135522027358 | 0.03777617298977736 | [0.04533532077366385, 0.24310131670556268] |
| r10 | (3, 10, 0.1) | 0.07019082640000009 | 0.9872048016 | 0.07019082640000009 | [0.0667395111777345, 0.6524528500599973] |
| r11 | (1, 10, 0.19) | 0.6989768703210189 | 0.4067564616548679 | 0.8784233454094306 | [0.0025285785444617865, 0.44501611702819543] |
| r12 | (10, 10, 0.05) | 9.765625000000033e-14 | 1.0 | 9.765625000000033e-14 | [0.6915028921812392, 1.0] |
| r13 | (95, 100, 0.5) | 1.2523245125385403e-22 | 1.0 | 6.261622562692701e-23 | [0.8871650888945373, 0.9835681208179479] |
| r14 | (7, 20, 0.3) | 0.6294979666766773 | 0.7722717974181605 | 0.39199018779907585 | [0.15390920478454115, 0.5921885345328282] |
| r15 | (12, 20, 0.3) | 0.005936084198097514 | 0.9987211203957798 | 0.005138161535121396 | [0.36054258730748967, 0.8088099392746928] |
| r16 | (2, 20, 0.3) | 0.05262794872972709 | 0.035483132298468674 | 0.9923627402258 | [0.012348527170294814, 0.3169827140190823] |
| r17 | (60, 100, 0.5) | 0.05688793364098087 | 0.9823998998911476 | 0.028443966820490434 | [0.4972091504223348, 0.6967052312971225] |
| r18 | (4, 25, 0.3) | 0.18827192967586837 | 0.09047191855413647 | 0.9667594834091572 | [0.04537945237170958, 0.36082845445927214] |
| r19 | (9, 25, 0.3) | 0.516560316985197 | 0.8105639764950546 | 0.3230718748971322 | [0.17971682058365537, 0.5747936504461506] |
| r20 | (0, 20, 0.05) | 0.6226463974646923 | 0.3584859224085422 | 1.0 | [0.0, 0.16843347098308536] |
| r21 | (30, 60, 0.4) | 0.11591191316361174 | 0.9555196858157825 | 0.07462369435142713 | [0.36806203194243675, 0.6319379680575632] |
| r22 | (18, 60, 0.4) | 0.14650251165689387 | 0.07187881730546673 | 0.9587117811878154 | [0.18845144541972264, 0.4320831338760617] |
| r23 | (40, 60, 0.4) | 3.3286834791723675e-05 | 0.9999914803784984 | 2.7047676241638936e-05 | [0.5331273252571694, 0.7831305545688299] |

**Rust test:** `stats::binomtest::tests::test_validate_binom_grid`

## Upper tails

Before, P(X >= x) was computed as 1 - cdf(x - 1), which cancels when the tail is small:

| Cell | R | p2a before | p2a after |
|---|---|---|---|
| r13 (95, 100, 0.5), greater | 6.261622562692701e-23 | 0 | 6.2616225626938803e-23 |
| r13, two-sided | 1.2523245125385403e-22 | 6.2616225626938803e-23 | 1.2523245125387761e-22 |
| r12 (10, 10, 0.05), greater | 9.765625000000033e-14 | 9.7699626167013776e-14 (relative error 4.4e-4) | 9.7656249999999827e-14 |

All after-values are within 2e-13 relative of R.

**Rust test:** `stats::binomtest::tests::test_validate_binom_upper_tails`

## STAT-15 (two-sided rule; fixed in Tier 4)

p2a started its search for the other tail at floor(np) and added P(X = 0) when no lower cell
qualified; R searches the whole side of the support beyond the mean. In nine cells the two-sided
p differed. They are now pinned under `p_two_stat15`; the values p2a gave before:

| Cell | (x, n, p) | R two-sided | p2a before |
|---|---|---|---|
| r04 | (3, 10, 0.37) | 0.753923870210605 | 1 |
| r05 | (7, 10, 0.75) | 0.7184324264526372 | 1 |
| r06 | (1, 10, 0.05) | 0.40126306076162127 | 0.99999999999999911 |
| r07 | (2, 10, 0.05) | 0.0861383558993165 | 0.68487529513769441 |
| r08 | (5, 100, 0.01) | 0.0034323215877545264 | 0.36946466286092494 |
| r09 | (6, 50, 0.05) | 0.03777617298977736 | 0.11472114826648963 |
| r10 | (3, 10, 0.1) | 0.07019082640000009 | 0.41886926650000395 |
| r11 | (1, 10, 0.19) | 0.6989768703210189 | 1 |
| r12 | (10, 10, 0.05) | 9.765625000000033e-14 | 0.59873693923847693 |

## Large n and x = np

Case `stat_binom_minlike_t4` (`validation/r/cases/stats_exact_t4.R`; dataset
`validation/datasets/generated/binom_minlike_t4.csv`), registry
`validation/methods/stats_exact_t4.json`, id `binom_minlike_t4`: the two-sided p at x = np and at
n = 1,000 and 10,000, each at `Tol::p_value(n)`.

| Cell | (x, n, p) | R two-sided p |
|---|---|---|
| m01 | (5, 10, 0.5) | 1 |
| m02 | (4950, 10000, 0.5) | 0.3221741997953122 |
| m03 | (5100, 10000, 0.5) | 0.04658552770494736 |
| m04 | (52, 1000, 0.04) | 0.06267066346387996 |
| m05 | (1, 10000, 5e-4) | 0.07218578443632245 |
| m06 | (20, 10000, 0.001) | 0.003934925807095321 |

**Rust test:** `stats::binomtest::tests::test_validate_binom_minlike_t4`

## Input checks

`p` and `conf_level` must lie strictly between 0 and 1 (`validate_level`; NaN included): before, a
NaN `conf_level` panicked in the Beta quantile and a NaN `p` was not refused. `n = 0` and `x > n`
are errors, as in R. Test: `stats::binomtest::tests::test_binom_test_rejects_invalid_level_and_p`.
