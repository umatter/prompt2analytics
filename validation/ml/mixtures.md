# Validation: mixture EMs (mixtools::normalmixEM, mvnormalmixEM; flexmix::flexmix)

## Method Overview

All in `crates/p2a-core/src/ml/advanced_clustering_mod/mixtures.rs`.

**`normal_mix_em`, `normal_mix_em_with`.** Univariate normal mixture EM. The log-likelihood includes the `2 pi` constant
(the former code left out `n/2 ln(2 pi)`). `seed = None` is a deterministic start, `Some(seed)` a seeded draw in
`normalmix.init`'s shape (the draw order is p2a's). The stop is `|l - l_prev| < tolerance` with mixtools' defaults (1000
iterations, `1e-8`). A collapsing component restarts the fit, as mixtools does, at most `MIXTOOLS_MAX_RESTARTS` times.
The M-step is standard EM; normalmixEM's sigma update uses the previous mu, which has the same fixed point.
Each sigma comes from distances to its new mu, so neither the data's location nor the gap between components costs
digits (`test_normal_mix_em_does_not_depend_on_the_data_location`, `test_normal_mix_em_variance_of_clusters_far_apart`).

**`mvnorm_mix_em`, `mvnorm_mix_em_with`.** Multivariate normal mixture EM with full covariances, on the shared
`gmm::{e_step, m_step}` (the former code fitted diagonal ones). Defaults 10000 iterations and `1e-8`; restarts at most
`MVMIXTOOLS_MAX_RESTARTS`.

**`flexmix`, `flexmix_with`.** A port of `flexmix::FLXfit` for Gaussian regression mixtures: components start from a
cluster vector (`FlexMixStart::Clusters`), a posterior matrix (`Posterior`) or a seeded draw (`Random`), `minprior`
(default 0.05) removes components whose prior falls below it at the start of an iteration, and the stop is the relative
change of the log-likelihood (`1e-6`). Each component's weighted fit uses the normal equations of the scaled design
first and the QR solve when they are singular, so a rank-deficient component is named. A component whose fit is exact
has `sigma` floored at `FLEXMIX_SIGMA_FLOOR_REL * sd(y)` and is reported in `degenerate_components` and a warning.

## Reference

`mixtools::normalmixEM`, `mixtools::mvnormalmixEM` (mixtools 2.0.0.1) and `flexmix::flexmix(y ~ x, model =
FLXMRglm(family = "gaussian"))` (flexmix 2.3.21) on R 4.3.3. The `flexmix` case also runs a written-out `FLXfit`, which
reproduces both configurations to 1e-12 (`stopifnot`).

## Where

R code `validation/r/cases/ml_mixtures.R`, cases `mixtools_em` (`mixture_3d.csv`) and `flexmix_minprior`
(`flexmix_lines.csv`, n = 200), under `validation/datasets/generated/`. Registry `validation/methods/ml_mixtures.json`,
methods `ml_mixtools` and `ml_flexmix`.

The `flexmix` case differs from the first design of the case: `minprior = 0.05` with `tolerance = 1e-12` and
`iter.max = 1000` prunes nothing and does not converge. The committed case runs `prune` (`minprior = 0.1`) and `keep`
(`minprior = 0.05`), both with `iter.max = 5000`, from one start.

## Values

**`mixtools_em`.**
- `uni_tight_loglik` -573.227084940149. The former code reported -297.5455249787472, this plus `150 ln(2 pi)`.
- `uni_tight_lambda` 0.41391918308784487, 0.22933754717637853, 0.3567432697357765.
- `uni_tight_mu` -0.1823747619685174, 1.28793968147256, 3.9437497101631944.
- `uni_tight_sigma` 0.8365247140706602, 0.5246442519567422, 0.684385441081849.
- `uni_loglik` (normalmixEM) -573.2270849401662.
- `uni_det_start_mu` -1.49764577875, 1.35819015655, 4.21402609185; `uni_det_start_loglik` -679.5810919056637.
- `mv_tight_loglik` -1442.6147212071523, the VVV optimum of `gmm_mclust_3d`.
- `mv_tight_lambda` 0.3950870265949487, 0.33773364882555107, 0.2671793245795005.
- `mv_loglik` (mvnormalmixEM) -1442.6147212071526.

**`flexmix_minprior`** (components ordered by intercept).

| | `prune` (minprior 0.1) | `keep` (minprior 0.05) |
|---|---|---|
| `k` / `k0` | 2 / 3 | 3 / 3 |
| `loglik` | -270.5475867607077 | -267.300928860085 |
| `df`, `aic`, `bic` | 7, 555.0951735214154, 578.1833950872517 | 11, 556.60185772017, 592.8833487521983 |
| `prior` | 0.6700584975074612, 0.3299415024925388 | 0.5992717233834256, 0.09017343276243842, 0.31055484385413595 |
| `sigma` | 0.632763000462263, 0.5511127241335194 | 0.5932778626664381, 0.3918287277727036, 0.49712076924501597 |

Iteration counts, the iteration of a removal and non-converged states are not R-pinned; they stay in the case JSON as
recorded values.

## Tolerances

- `EM_PARAM` 1e-5 relative and `EM_LOGLIK` 1e-12 relative against the tight values, as in `gmm.md`: `|dl| < 1e-12`
  leaves the univariate parameters 1.69e-6 from the optimum and `|l - l*|` at 2.1e-11.
- `FIXED_POINT` 1e-10 relative, started at the tight values.
- flexmix `prune` 1e-9 relative (same algorithm, same stop; the last parameter step is 6.7e-10). `keep` 1e-7 relative
  and log-likelihood 1e-11: the stop is borderline, one iteration moves the parameters 2.2e-9.
- `k`, `k0`, `df`, clusters exact.

## Known differences

- `normal_mix_em`: standard EM against normalmixEM's previous-mu sigma update (same fixed point).
- `normalmix_init`: the redraw of `lambda` is bounded (R's loop does not end for k > 20: the limit is `min(0.05,
  1/(2k))` here, and equal weights after 100 draws).
- flexmix exact-fit component: floor and warning where R stops with "Log-likelihood: NaN" (`stopifnot` in the case).
- flexmix's random start uses p2a's RNG, so the pins pass `FlexMixStart::Clusters` with the case's start column.

## Tests

R-pinned: `test_validate_mixtools_uni`, `test_validate_mixtools_uni_start`, `test_validate_mixtools_uni_fixed_point`,
`test_validate_mixtools_mv`, `test_validate_mixtools_mv_start`, `test_validate_flexmix_minprior`,
`test_validate_flexmix_keep`.

Properties: `test_normal_mix_em_uses_its_seed`, `test_normalmix_init_draws_valid_starts_for_any_k`,
`test_mixture_em_stopping_rule_is_scale_free`, `test_normal_mix_em_restarts_when_a_variance_collapses`,
`test_mvnorm_mix_em_default_seed_is_reproducible`, `test_mixtools_rejects_bad_parameters`,
`test_flexmix_flags_degenerate_components`, `test_flexmix_exact_fit_is_floored_and_flagged`,
`test_flexmix_rejects_bad_parameters`, `test_flexmix_rank_deficient_component_is_refused`.

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- advanced_clustering_mod::mixtures::tests::test_validate
```
