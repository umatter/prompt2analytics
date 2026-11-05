# Econometrics Validation

This directory contains validation documentation for econometric methods.

## Methods

### Panel Data
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Fixed Effects | [panel_fe.md](panel_fe.md) | `run_fixed_effects()` | R `plm` (within) |
| Random Effects | [panel_re.md](panel_re.md) | `run_random_effects()` | R `plm` (random) |
| Hausman Test | [hausman.md](hausman.md) | `run_hausman_test()` | R `plm::phtest()` |
| HDFE | [hdfe.md](hdfe.md) | `run_hdfe()` | R `lfe::felm()` |
| Arellano-Bond difference GMM | [gmm.md](gmm.md) | `run_gmm()` | R `plm::pgmm()` |
| Panel unit-root tests | [panel_unit_root.md](panel_unit_root.md) | `run_panel_unit_root()` | R `plm::purtest()` |

### Instrumental Variables
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| 2SLS | [iv_2sls.md](iv_2sls.md) | `run_iv2sls()` | R `AER::ivreg()` |

### Causal Inference
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Diff-in-Diff | [did.md](did.md) | `run_did()` | Manual calculation |
| IPW Treatment Effects | [treatment_ipw.md](treatment_ipw.md) | `run_ipw_treatment()` | R `causalweight::treatweight()` |
| Doubly Robust (AIPW) | [treatment_aipw.md](treatment_aipw.md) | `run_doubly_robust()` | R `causalweight`, `AIPW` |
| Mediation Analysis | [mediation.md](mediation.md) | `run_mediation_analysis()` | R `causalweight::medweight()` |
| Regression Discontinuity (sharp, fuzzy) | [rd.md](rd.md) | `run_rd()`, `run_fuzzy_rd()` | R `rdrobust::rdrobust()` |

### Discrete Choice
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Logit | [logit.md](logit.md) | `run_logit()` | R `glm()` |
| Probit | [probit.md](probit.md) | `run_probit()` | R `glm()` |
| Ordered Logit and Probit | [ordered.md](ordered.md) | `run_ordered_logit()`, `run_ordered_probit()` | R `MASS::polr()` |
| Count Models (negative binomial, ZIP, ZINB) | [count_models.md](count_models.md) | `run_negbin()`, `run_zip()`, `run_zinb()` | R `MASS::glm.nb()`, `pscl::zeroinfl()` |
| Hurdle Models | [hurdle.md](hurdle.md) | `run_hurdle()` | R `pscl::hurdle()` |
| Mixed Logit | [mixed_logit.md](mixed_logit.md) | `run_mixed_logit()`, `run_mixl()`, `run_gmnl()` | R `mlogit::mlogit()` (random parameters) |

### Time Series
See [timeseries/](timeseries/) subdirectory.

| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Granger causality F test | [granger.md](granger.md) | `granger_test()` | R `lmtest::grangertest()` |

### Survival Analysis
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Kaplan-Meier | [survival.md](survival.md) | `run_kaplan_meier()` | R `survival::survfit()` |
| Log-Rank Test | [survival.md](survival.md) | `log_rank_test()` | R `survival::survdiff()` |
| Cox PH | [survival.md](survival.md) | `run_cox_ph()` | R `survival::coxph()` |
| AFT Models | [survival.md](survival.md) | `run_aft()` | R `survival::survreg()` |
| Competing Risks | [survival.md](survival.md) | `run_competing_risks()` | R `survival::survfit()` (multi-state) |

## Key Test Datasets

- **Grunfeld (1958)**: Panel data for firm investment (n=200, 10 firms × 20 years)
- **Synthetic panel data**: Known DGP with controlled fixed effects

## Running Tests

```bash
# All econometrics validation tests
cargo test -p p2a-core -- econometrics::tests::test_validate

# HDFE specifically
cargo test -p p2a-core -- hdfe::tests::test_validate
```
