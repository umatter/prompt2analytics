# Econometrics Validation

Validation documents for the econometric methods in `crates/p2a-core/src/econometrics/`. Whether
each is pinned to R, and with which known deviations, is in the method index of
[../README.md](../README.md) (derived from `validation/methods/*.json`) and on
`validation/VALIDATION_STATUS.md`; the reference column names the R function a document's case
uses, or the formula written out in R where no package is installed.

## Methods

### Panel data
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Fixed effects (within) | [panel_fe.md](panel_fe.md) | `run_fixed_effects()` | R `plm::plm(model = "within")` |
| Random effects (Swamy-Arora) | [panel_re.md](panel_re.md) | `run_random_effects()` | R `plm::plm(model = "random")` |
| Hausman test | [hausman.md](hausman.md) | `run_hausman_test()` | R `plm::phtest()` |
| High-dimensional fixed effects | [hdfe.md](hdfe.md) | `run_hdfe()` | R `fixest::feols()` |
| Panel GLS | [pggls.md](pggls.md) | `run_panel_gls()`, `run_pooled_gls()` | R `plm::pggls()` |
| Variable coefficients and mean group | [pvcm.md](pvcm.md) | `run_pvcm()`, `run_pmg()` | R `plm::pvcm()` |
| Arellano-Bond and Blundell-Bond GMM | [gmm.md](gmm.md) | `run_gmm()`, `run_arellano_bond()` | R `plm::pgmm()` |
| FEGLM | [feglm.md](feglm.md) | `run_feglm()` | R `fixest::feglm()` |
| Panel unit-root tests | [panel_unit_root.md](panel_unit_root.md) | `run_panel_unit_root()` | R `plm::purtest()` |

### Instrumental variables, GMM and bounds
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| 2SLS | [iv_2sls.md](iv_2sls.md) | `run_iv2sls()` | R `AER::ivreg()` |
| General GMM | [general_gmm.md](general_gmm.md) | `run_gmm_iv()`, `run_general_gmm()` | R `gmm::gmm()` |
| Marginal treatment effects | [ivmte.md](ivmte.md) | `run_ivmte()` | Heckman-Vytlacil formulas written out in R |
| Balke-Pearl bounds | [bpbounds.md](bpbounds.md) | `run_bp_bounds()` | closed form, hand-computed tables (bpbounds not installed) |

### Difference-in-differences
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Canonical 2x2 DiD | [did.md](did.md) | `run_did()` | R `stats::lm()` with `sandwich` |
| Staggered DiD (Callaway-Sant'Anna) | [staggered_did.md](staggered_did.md) | `run_staggered_did()` | R `did::att_gt()`, `DRDID` |
| Extended two-way fixed effects | [etwfe.md](etwfe.md) | `run_etwfe()` | R `etwfe::etwfe()`, `emfx()` |
| Goodman-Bacon decomposition | [bacon.md](bacon.md) | `bacon_decomp()` | R `bacondecomp::bacon()` |

### Regression discontinuity
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Sharp and fuzzy RD, bandwidth selection | [rd.md](rd.md) | `run_rd()`, `run_fuzzy_rd()` | R `rdrobust::rdrobust()`, `rdbwselect()` |
| Multi-cutoff RD | [rdmulti.md](rdmulti.md) | `run_rd_multi()` | R `rdrobust` per cutoff; the pooling written out in R |

### Weighting and matching
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| IPW | [treatment_ipw.md](treatment_ipw.md) | `run_ipw_treatment()` | R `glm(binomial)` propensity, `WeightIt` |
| Doubly robust (AIPW) | [treatment_aipw.md](treatment_aipw.md) | `run_doubly_robust()` | R `lm` outcome models, `glm(binomial)` propensity |
| WeightIt and entropy balancing | [weightit.md](weightit.md) | `weightit()`, `entropy_balance()` | R `WeightIt::weightit()` |
| CBPS | [cbps.md](cbps.md) | `run_cbps()` | R `CBPS::CBPS()` |
| Stable balancing weights | [sbw.md](sbw.md) | `run_sbw()` | Zubizarreta (2015) QP written out in R (`quadprog`) |
| GBM propensity scores (twang) | [twang.md](twang.md) | `run_twang()` | none (twang not installed) |
| Matching | [matching.md](matching.md) | `match_it()`, `full_match()` | R `MatchIt::matchit()`, `optmatch::fullmatch()` |

### Targeted learning and g-methods
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| TMLE | [tmle.md](tmle.md) | `run_tmle()` | R `tmle` 2.1.1's algorithm with `glm` fits |
| Collaborative TMLE | [ctmle.md](ctmle.md) | `run_ctmle()` | Gruber and van der Laan (2010) criteria written out in R |
| LTMLE | [ltmle.md](ltmle.md) | `run_ltmle()` | R `ltmle` |
| Regression standardisation (stdReg) | [stdreg.md](stdreg.md) | `run_stdreg()` | R `glm` and standardisation as `stdReg::stdGlm` (written out) |
| Parametric g-formula | [gformula.md](gformula.md) | `run_gformula()` | g-formula written out in R |
| Double/debiased ML | [doubleml.md](doubleml.md) | `run_double_ml()`, `run_double_ml_iv()` | DoubleML's PLR, IRM, PLIV and IIVM scores written out in R |

### Synthetic control
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Synthetic control | [synth.md](synth.md) | `run_synthetic_control()` | the Synth QP solved by `quadprog::solve.QP` |
| Generalized synthetic control | [gsynth.md](gsynth.md) | `run_gsynth()` | R `gsynth::gsynth()` (`fect`) |
| SCPI | [scpi.md](scpi.md) | `run_scpi()` | p2a's formulas written out in R |

### Mediation
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Causal mediation (IPW) | [mediation.md](mediation.md) | `run_mediation_analysis()` | R `glm(binomial)` propensity scores and the Hajek total effect |
| Natural effect models | [medflex.md](medflex.md) | `run_medflex()` | R `medflex::neImpute()`, `neModel()` |

### Discrete choice
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Logit | [logit.md](logit.md) | `run_logit()` | R `glm(binomial("logit"))` |
| Probit | [probit.md](probit.md) | `run_probit()` | R `glm(binomial("probit"))` |
| Conditional and multinomial logit | [mlogit.md](mlogit.md) | `run_mlogit()`, `run_conditional_logit()` | R `mlogit::mlogit()`, `nnet::multinom()` |
| Ordered logit and probit | [ordered.md](ordered.md) | `run_ordered_logit()`, `run_ordered_probit()` | R `MASS::polr()` |
| Newton loops of multinomial and ordered models | [discrete_newton.md](discrete_newton.md) | `run_multinom()`, `run_ordered_logit()` | R `nnet::multinom()`, `MASS::polr()`, polished |
| Count models (negative binomial, ZIP, ZINB) | [count_models.md](count_models.md) | `run_negbin()`, `run_zip()`, `run_zinb()` | R `MASS::glm.nb()`, `pscl::zeroinfl()` |
| Hurdle models | [hurdle.md](hurdle.md) | `run_hurdle()` | R `pscl::hurdle()` (zero part as a binomial `glm`) |
| Mixed logit | [mixed_logit.md](mixed_logit.md) | `run_mixed_logit()`, `run_mixl()`, `run_gmnl()` | R `mlogit::mlogit()` (random parameters) |

### Survival
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Kaplan-Meier, log-rank, Cox PH, AFT, competing risks | [survival.md](survival.md) | `run_kaplan_meier()`, `log_rank_test()`, `run_cox_ph()`, `run_aft()`, `run_competing_risks()` | R `survival::survfit()`, `survdiff()`, `coxph()`, `survreg()` |

### Spatial
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| SAR, spatial Durbin, SEM, SAC | [spatial.md](spatial.md) | `run_sar()`, `run_sem()` | R `spatialreg::lagsarlm()`, `errorsarlm()` |
| Spatial GMM | [sphet.md](sphet.md) | `run_sphet()` | Kelejian-Prucha GM and spatial HAC (`sphet`) |
| Spatial panel models | [splm.md](splm.md) | `run_spml()`, `run_spgm()` | `splm`'s likelihoods and moments written out in R |
| Spatial probit | [spatialprobit.md](spatialprobit.md) | `run_sar_probit()`, `run_sem_probit()` | none |

### Time series
| Method | File | p2a Function | Reference |
|--------|------|--------------|-----------|
| Granger causality F test (integer series, missing values) | [granger.md](granger.md) | `granger_test()` | R `lmtest::grangertest()` |
| Multivariate time series (VAR, IRF, Granger, VARMA, VECM) | [timeseries/README.md](timeseries/README.md) | see the index | `vars`, `lmtest`, `urca`, Hannan-Rissanen written out |

## Key Test Datasets

- **Grunfeld (1958)**: Panel data for firm investment (n=200, 10 firms × 20 years)
- **Simulated data**: the generated datasets of the R cases, listed in
  [../datasets/README.md](../datasets/README.md)

## Running Tests

```bash
validation/run_validation.sh --rust-only --category econometrics
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::hdfe::tests::test_validate
```
