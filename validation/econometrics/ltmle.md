# LTMLE (Longitudinal Targeted Maximum Likelihood Estimation) Validation

## 1. Method

`run_ltmle` / `ltmle` (`crates/p2a-core/src/econometrics/ltmle.rs`) estimate the mean outcome
`E[Y^d]` under a treatment regime `d = (d_1, ..., d_T)` over `T >= 2` time points with
time-varying confounding (nodes `L_1, A_1, ..., L_T, A_T, Y`), and the contrast between two
regimes, by the longitudinal TMLE of the R package `ltmle` 1.x (Lendle et al. 2017) with its
defaults: `stratify = FALSE`, main-term `glm` models on the full history, `gbounds` on the
cumulative probability of following the regime, and the influence-curve variance.

## 2. Reference implementation

The reference is the estimator written out in R with `glm` and `lm`
(`validation/r/cases/causal_ltmle.R`), so that every intermediate (H, cumulative g, epsilons, Q
and g coefficients) can be pinned. The case cross-checks it against `ltmle::ltmle` 1.3-0
itself: the logistic-Q regime means and EIC standard errors of the two-period binary case
(always, never, dynamic), of `gbounds = c(0.1, 0.5)`, of the rescaled continuous outcome and of
the three-period case (always, never, dynamic, (1, 0, 1)) must agree to 1e-8 relative, or the
case stops (`ltmle(.., SL.library = "glm", stratify = FALSE, variance.method = "ic")`; ltmle
fits at glm's default control, about 1e-11 from the tight fits). R 4.3.3; `packages = {ltmle}`.
The linear Q model is p2a's own variant and has no ltmle counterpart: it is pinned to the
written-out reference only. On a regime with a handful of followers the written-out estimator
stops solving the EIC equation, and the case `ltmle_saturation` pins `ltmle::ltmle` itself
("Saturating regimes" below).

## 3. Algorithm as implemented

Nodes are ordered `L_1, A_1, L_2, A_2, ..., L_T, A_T, Y`; time points are 1-based in messages.

1. **History designs.** The g design of node t has the columns `(Intercept)`, then for every
   earlier node its covariates and its treatment, then the covariates of node t (R's
   `reformulate` order: `A2 ~ L0 + A1 + L1`). The Q design appends `A_t`
   (`Y ~ L0 + A1 + L1 + A2`). The designs are built once per call.
2. **g.** For each node a binomial logit GLM (glm.fit's IRLS; see "Nuisance fits" below) of
   `A_t` on its g design, on all subjects. Its fitted values are reported as
   `propensity_scores` (unbounded).
3. **Cumulative g (`CalcCumG`, `SetA`).** g is predicted with every earlier `A_s` set to the
   regime; the probability of the regime's treatment is `p` or `1 - p`; the cumulative product
   `c_t` is bounded to `gbounds` (default `(0.01, 1)`), not each factor. `n_bounded` counts the
   subjects outside the bounds, `n_bounded_following` those among the regime's followers.
4. **Clever covariate.** `H_t = I(Abar_t = dbar_t) / c_t` (0 for non-followers).
5. **Backward pass (`UpdateQ`).** From `t = T` to 1, starting from `Q*_{T+1} = Y`: a
   quasibinomial logit GLM of `Q*_{t+1}` on the Q design (pooled over all subjects), predicted
   on the link scale with every `A_s, s <= t`, set to the regime; the fluctuation is the
   intercept-only quasibinomial fit with offset `logit Q_t(d)` and weights `H_t` (zero weights
   drop the non-followers); `Q*_t = plogis(logit Q_t(d) + eps_t)` for every subject. The last
   node's Q regression has the same response and design for every regime and is fitted once.
   A Q response whose range over the subjects is below 1e-4 is not regressed: as
   `ltmle:::Estimate`, the node predicts `max(response)` for every subject, on the link scale
   `qlogis` of it bounded to `[1e-4, 0.9999]` (ltmle's no-variation rule;
   `NodeModel::constant_prediction`). The fluctuation is LTMLE's own scalar Newton solver of the
   score equation with glm.fit's deviance rule (CAUS-101); after it the node's EIC term
   `sum_i H_t,i (Q*_{t+1,i} - Q*_t,i)` is checked against ltmle's trigger of 1e-3 (absolute, on the
   fitted [0, 1] scale), and above it the score equation is re-solved with the offsets bounded to
   `logit([1e-8, 1 - 1e-8])` (`ltmle:::FixScoreEquation`'s first bound), by a bracketed monotone
   root search to `1e-12 * sum_i H_t,i` instead of ltmle's nlminb to 1e-4. With responses neither
   all at 0 nor all at 1 the score changes sign in the bracket, so this search always ends at a
   root, or at a bracket at rounding, on that first bound; ltmle's later bounds (1e-4, 1e-3, 0.01),
   which it tries when nlminb fails, are kept only for the search's iteration limit. With every
   response at 0 (or 1) the score has no root and every follower is predicted at the link's clamp.
6. **Estimate and EIC (`CalcIC`).** `psi = mean(Q*_1)`,
   `EIC = sum_t H_t (Q*_{t+1} - Q*_t) + Q*_1 - psi`.

Aliased columns (a duplicated history column, say) get NaN coefficients and predictions use the
identified columns, as `predict.glm`; the aliasing, convergence and separation warnings of every
fit are reported with the model they come from.

### Nuisance fits

Every logistic g and Q regression first runs `econometrics::glm::lean_logit_refined`: the same
Fisher scoring and deviance rule as `fit_glm`, through the normal equations, with the refinement
step the shared GLM takes when `kappa_1` of the equilibrated `X'WX` is above 100, on the steps
near the estimate (the previous step moved the deviance by less than 1e-3 relative, or the fit
started from coefficients). A step the deviance rule stops at without one takes it afterwards,
from the fit it started at. It computes only what LTMLE reads. A fit leaves for `fit_glm` from
glm.fit's start, which then reports it, on a linear predictor at the link's clamp, a rank-deficient
design or `kappa_1` above 1e4, no convergence, a binomial response that is not 0/1, or a last step
that no longer meets the rule once refined. A converged fit also leaves with a row beyond
`|eta| = 15` on the wrong side (`y < 1` above 15, `y > 0` below -15), because on a 0/1 response
`fit_glm` loses the digits of `1 - mu` there and can report the fit differently. The check is the
same for a fractional Q response, where `1 - 1e-8` at `eta = 18` leaves although it is well fitted;
the bound was calibrated on 0/1 responses only, and on the sweep below no fast-path fit of 23,104
left through it at 1e-8 and 2 of 23,102 at 1e-10. A Q response with a range below 1e-4 is not
fitted at all (ltmle's no-variation rule, step 5 above), on every path alike. A response whose
mean is within 1e-6 of 0 or 1 and whose range is at least 1e-4 goes to `fit_glm` directly. On the
sweep below, at the default tolerance, the rule takes 5,449 Q fits of 520 data sets off the shared
GLM; the shared-GLM warnings of Q fits fall from 4,556 to 22, the data sets reported as not
converged from 371 to 0, and the iterative fits from 48,934 to 43,466 ("Saturating regimes").

The Q regression of node `t - 1` starts from node t's coefficients on the shared history, with
`L_t`, `A_t` and `eps_t` moved into the intercept at their means (`warm_start`); its response is
node t's targeted prediction. The coefficients of the earlier `A_s` stay on the observed columns,
although that response has those treatments at the regime's values, so the start is approximate
and IRLS corrects it. The fit takes glm.fit's start instead when `max_iter` is below 50: the limit
decides what is reported as not converged, on the step count from glm.fit's start, which a warm
start would undercut. Of the 11,292 Q fits with a response mean off the 1e-6 guard in the sweep
below, counting all of them at the default tolerance, the most any took from glm.fit's start is
42, and the most among those whose warm and cold step counts differ is 16. A response mean within
1e-6 of 0 or 1 never takes the warm path. The warm start is not always the shorter: 589 of those
fits took more steps from it, the most being 86 where glm.fit's start took 12
(`shifted_outcome((4000, 12, 1), 4.0, true, 0.3)`, node 7); at a limit below such a count the
lean fit leaves for `fit_glm` from glm.fit's start. At `max_iter` 50, 60 and 75 the warnings and
convergence status are equal on all 1,716 data sets.

The gap table and figures below were measured at bd5068dd, before the no-variation rule and the
score re-solve; the columns with a regime near 0 or 1 describe fits that the rule now skips.

Two reported things change, and both are accepted. `NodeModel::iterations` of a warm-started node
counts the steps from its warm start (2 where glm.fit's start takes 4 or 5), so it is not comparable
with glm.fit's.
And each fit stops at the deviance rule on a path that depends on its start, so at the default
tolerance the numbers move by the rule's slack. Fast paths against `fit_glm` from glm.fit's start,
at tolerances 1e-8 (the default), 1e-10 and 1e-14, on 2,160 data sets of which 1,716 fit (both
paths refuse the same 444): `long_horizon` and `sparse_followers` with T in {3, 4, 5, 6, 8, 10, 12}, n in {150, 300, 600, 1200, 2500} and 24 seeds, and `shifted_outcome`
with 12 (n, T) pairs from (150, 4) to (5000, 10) and (4000, 12), eight (shift, outcome kind,
confounding) settings with shifts from -6 to 4, binary and fractional outcomes, and 5 seeds. Each
gap is the largest over the data sets, relative to the largest entry of the same vector. On 523
data sets a regime's mean is within 1e-6 of 0 or 1 at the default tolerance; they have their own
columns. "Warm off" is the lean path with the warm start off, the same set of data sets:

| Quantity | tolerance 1e-8 | warm off | 1e-8, regime mean within 1e-6 of 0 or 1 | warm off | tolerance 1e-14 | near-duplicate at 1e-7 |
|---|---:|---:|---:|---:|---:|---:|
| ATE | 5.3e-10 | 1e-13 | 2.1e-10 | 5e-12 | 6e-12 | 2e-11 |
| z | 5.3e-10 | 3e-13 | 4.6e-10 | 5e-12 | 1.1e-12 | 3e-11 |
| SE | 3.6e-11 | 3e-13 | 3.2e-10 | 8e-13 | 6e-12 | 3e-11 |
| regime means | 1.2e-10 | 5e-15 | 1.8e-7 | 1.8e-7 | 6e-12 | 2e-11 |
| regime SE | 7.4e-11 | 3e-12 | 1.9e-8 | 1.9e-8 | 6e-12 | |
| p-value | 8.1e-10 | 2e-10 | 5.0e-9 | 2e-12 | 2e-11 | 1e-9 |
| EIC of the ATE | 1.2e-10 | 3e-12 | 8.8e-10 | 2e-12 | 6e-12 | 2e-10 |
| EIC of a regime | 1.3e-10 | 1e-11 | 8.2e-8 | 8.2e-8 | 9e-12 | |
| Q* | 1.7e-9 | 1e-13 | 3.6e-7 | 3.6e-7 | 2e-11 | 1e-9 |
| Q coefficients | 1.2e-9 | 1e-12 | 7.3e-9 | 7.3e-9 | 1.3e-10 | 4e-9 |
| fluctuation `eps` | 1.2e-8 | 4e-10 | 8.6e-8 | 6.7e-9 | 9e-11 | 7e-10 |
| g coefficients, propensities, cumulative g, H | 6e-15 | 6e-15 | 6e-15 | 6e-15 | 5e-14 | 1e-15 |

The near-duplicate column is one fixture family, not this sweep; its two empty rows were not
measured separately. These gaps are accepted, including 5.3e-10 on the ATE and z and 1.8e-7 to
3.6e-7 on regime means and Q* with a regime near 0 or 1. Where they come from:

- **The warm start, on ordinary data.** At the default tolerance the warm-off columns show the lean
  path keeps `fit_glm`'s numbers to at most 5e-12 on the ATE, SE and z. The gap above that is the
  warm start's, and of the size of the deviance rule's slack, not an accuracy loss: against
  `fit_glm` at 1e-14 on the 1,193 data sets with no regime near 0 or 1, the fast path's ATE is the
  closer on 590 and `fit_glm`'s at 1e-8 on 540 (63 tied). At `shifted_outcome((300, 3, 1), 1.5,
  false, 1.2)` (ATE 0.003, gap 4.6e-10) the fast path is 2.7e-11 from that reference and `fit_glm`
  at 1e-8 is 4.9e-10; at `long_horizon(300, 6, 17)` (gap 1.7e-10) they are 1.8e-11 and 1.8e-10. It
  is the rule's slack in size, not bounded by the slack data set by data set: on 224 of the 1,193
  the ATE's gap is larger than `fit_glm`'s own distance to the reference, the most at
  `sparse_followers(150, 6, 0)` (3.8e-11 absolute against 1.1e-12).
- **An always-near-0 regime, without the warm start.** At `long_horizon(1200, 8, 21)` the never
  regime's mean is about 1.5e-13 and most of its fits run the 100 steps without converging. The
  lean path and the fast path are bit-identical there; they move psi by 1.8e-7 relative (2.8e-20
  absolute). The largest Q* and the other regime rows are at `sparse_followers(1200, 8, 17)`, with a
  regime mean of 1.0e-13: Q* by 3.6e-7 relative (5.5e-20 absolute), regime SE by 1.9e-8, a regime's
  EIC by 8.2e-8 and the Q coefficients by 7.3e-9.
- **Small denominators.** A relative gap is large where its reference is small: the 5.3e-10 on
  the ATE is 5.9e-13 absolute, at `shifted_outcome((150, 4, 3), 1.5, false, 1.2)`; the largest ATE
  gap in absolute terms is 5.2e-11. The vector scale also hides small entries: on its own scale one
  eps entry moves by 9e-5 (`shifted_outcome((1000, 3, 4), -5.0, false, 0.3)`). The other way, where
  a flat likelihood sends an entry far out, a large absolute move is small on the vector's scale:
  an eps of order 1e14 by 2.5 at `long_horizon(150, 5, 8)`, a Q coefficient of order 1e15 by 1.09e7 at
  `long_horizon(1200, 8, 16)`. At tolerance 1e-10 such a coefficient moves by 2.3e-2 of the
  largest entry (`shifted_outcome((300, 8, 3), -1.0, false, 0.3)`, warm off the same). At 1e-10
  only Q coefficients go above 1e-9, on four data sets, and the estimates stay at 3.5e-10 or
  below. The 1e-14 column's 1.3e-10 is the same kind of entry (`long_horizon(300, 8, 18)`; the largest with
  no regime near 0 or 1 is 5e-14).

So at the default tolerance neither 1e-10 on the ATE, z and SE nor 1e-9 on the nuisance quantities
is a general bound: 70 of the 1,716 data sets exceed one of them
(46 with no regime near 0 or 1). The tests assert those two bounds, 1e-10 on the ATE, SE, z and
regime means and 1e-9 on the rest at the default tolerance, on their own fixtures only, plus 1e-10
on every quantity at 1e-14 and 1e-8 on the near-duplicate.
`test_ltmle_fast_paths_hold_the_gaps_measured_off_the_fixtures` runs the largest cases above, each
with the bound it meets, and checks that the gap is the warm start's where this section says so.
The near-duplicate's gap is the same with the warm start off: the condition number of about 1e7
amplifies the rounding that reaches the clever covariate from the g fits, on the near-duplicate's
coefficient and the predictions. On a saturated node the likelihood is flat, and those fits run
`fit_glm` from glm.fit's start.

At tolerances 1e-8 and 1e-10 the warnings and convergence status are equal on all 1,716 data sets.
At `max_iter` 2 to 6 and 49 they are equal by construction. At 1e-14, where rounding decides the
deviance rule, the status differs on 158 of the 1,716 and the warnings on 305, with the numbers
within the 1e-14 column. No data set fits on one path and errors on the other.

The sweep also showed something the fast paths did not change: on a regime that saturated, the
estimate itself depended on the tolerance, on both paths alike (`shifted_outcome((5000, 10, 4),
3.0, false, 2.0)`: always-treat mean 2e-16 at 1e-8 and 1 at 1e-10; `sparse_followers(2500, 8, 20)`:
never-treat mean 9.4e-3 at 1e-8 and 2e-16 at 1e-10). The follow-up settled it against
`ltmle::ltmle` 1.3-0: ltmle's no-variation rule and score check, which p2a lacked, are what keep
ltmle stable at its own glm epsilon of 1e-8. With both in place the two data sets agree with ltmle
within 1e-8 at every tolerance; "Saturating regimes" has the measurements.

There is no warm start across regimes. The last node's Q regression is fitted once for all
regimes, and every earlier one already starts from its own regime's later node and takes about
two steps, so a start from the other regime has few steps left to save. The fits that stay
expensive are the saturated ones, which keep glm.fit's start ("Wall-clock against fe13b889").

### Saturating regimes

A regime with a handful of followers at the last node (10 or fewer of thousands) has a
targeting step that drives the later Q* towards 0 or 1. Measured on the sweep's 2,160 data sets
(the fast-path grid above) at tolerances 1e-8, 1e-9, 1e-10, 1e-12 and 1e-14, before
and after the two rules below, with `ltmle::ltmle` 1.3-0 (`SL.library = "glm"`, `stratify =
FALSE`, `variance.method = "ic"`) run on the exported data sets as the oracle. Means on the
probability scale:

| Data set (followers of always / never at the last node) | always | never | ATE |
|---|---:|---:|---:|
| `shifted_outcome((5000, 10, 4), 3, false, 2)` (10 / 7): ltmle | 0.99999999995095079 | 0.99999999994364086 | 7e-12 |
| p2a before, tolerance 1e-8 (not converged) / 1e-14 | 2.2e-16 / 1 | 1 / 1 | -1 / 0 |
| p2a after, 1e-8 / 1e-14 | 0.9999999999999998 / 0.9999999999999998 | 0.99999999999984 / 0.9999999999999998 | 1.6e-13 / 0 |
| written-out glm estimator, epsilon 1e-8 / 1e-14 | 0.99999999999925 / 0.99896149 | 0.99999999999912 / 0.99998683 | |
| analytic estimand of the DGP | 0.999036 | 0.950235 | 0.048802 |
| `sparse_followers(2500, 8, 20)` (19 / 10): ltmle | 0.74648582426651022 | 4.0282706210332339e-11 | 0.746485824226 |
| p2a before, 1e-8 / 1e-14 | 0.746485824267 / 0.746485824248 | 0.0093744942864 / 2.2e-16 | 0.737111329980 / 0.746485824248 |
| p2a after, 1e-8 / 1e-14 | 0.746485824267 / 0.746485824248 | 2.2e-16 / 2.2e-16 | 0.746485824267 / 0.746485824248 |
| written-out, epsilon 1e-8 / 1e-14 | 0.74648582426656 / 0.74648582424775 | 7.9e-13 / 0.0048317 | |
| analytic estimand | 0.826621 | 0.157888 | 0.668733 |
| `sparse_followers(4000, 10, 3)` (9 / 4), the timing harness's saturating case: ltmle | 0.89052550172634615 | 0.49401994965066831 | 0.39650555207568 |
| p2a before, every tolerance | 0.8905255021 | 1 | -0.1094744979 |
| p2a after, 1e-8 / 1e-14 | 0.890525502108 / 0.890525502108 | 0.494019949648 / 0.494019949647 | 0.396505552461 / 0.396505552461 |

What the runs show:

1. **A tight tolerance is not a reference on these data.** The written-out glm estimator (the
   reference of `ltmle_two_period`) stops being monotone in epsilon: at epsilon 1e-14 its
   fluctuation coefficients reach 4.5e15 (on the three data sets named here), and its EIC mean at epsilon 1e-14 is -0.0053 (sparse 2500, never), -0.069
   (`shifted_outcome((150, 4, 2), 3, false, 2)`, never) and 0.063 (`sparse_followers(150, 8, 5)`,
   never): its targeting does not solve the EIC equation there. The criterion is that every
   node's score equation is solved, not a tolerance.
2. **ltmle stays stable because of two rules, which p2a now applies** (step 5 above): the
   no-variation rule (a Q response with a range below 1e-4 is predicted as a constant, before
   the choice of fit path) and the score check with its re-solve. Without the first, the fit of
   such a response ran to the link's clamp and its regime predictions depended on where the
   deviance rule stopped. Without the second, the scalar fluctuation could stop on its deviance
   rule far from the root: on `sparse_followers(4000, 10, 3)` the never regime's last step left
   the score at -200 and the mean at 1, at every tolerance.
3. **No estimator recovers the analytic estimand with about 10 followers** (never: 0.158 against
   about 0 for all of them). With so few followers the regime's mean is not identified in
   practice. ltmle warns that the influence-curve variance "may be significantly
   anticonservative because your data appears to contain positivity violations", and p2a now
   does too (warnings below). "Right" here means the ltmle oracle's value plus that warning.

**Tolerance dependence after the rules** (ATE at each tolerance against 1e-14, absolute, over the
1,715 data sets that fit; a constant outcome is now refused, which removes one):

| ATE moved by more than | 1e-6 | 1e-3 | 1e-2 | 0.1 | 0.5 |
|---|---:|---:|---:|---:|---:|
| before, tolerance 1e-8 | 17 | 9 | 6 | 3 | 2 |
| before, 1e-10 | 13 | 8 | 5 | 0 | 0 |
| after, 1e-8: flagged (914) / unflagged (801) | 5 / 0 | 2 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| after, 1e-9: flagged / unflagged | 4 / 0 | 2 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| after, 1e-10: flagged / unflagged | 4 / 0 | 3 / 0 | 2 / 0 | 0 / 0 | 0 / 0 |

"Flagged" is a data set with any saturation warning (mostly ltmle's positivity check). On the
unflagged ones the largest move between 1e-8 and 1e-14 is 9.4e-10. Of the 17 data sets that moved
before, 12 had a regime with a Q response of range below 1e-4 (three of them also an unsolved
score); the no-variation rule settles all 12. The other five still move, at all tolerances alike
(largest 0.023, `shifted_outcome((150, 4, 2), 3, false, 2)`): their last node's Q regression is
quasi-separated on a 0/1 outcome (30 to 35 IRLS steps, one with 12), and the regime prediction
from it depends on where the deviance rule stops. Neither rule addresses that, and a tighter
default does not either (the same five at 1e-10 and 1e-12).

**Agreement with ltmle after the rules** (|p2a - ltmle| on the regime means and the ATE):

| Oracle set | data sets | before, 1e-8 | after, 1e-8 | after, 1e-14 |
|---|---:|---:|---:|---:|
| regular (no regime near 0 or 1, no move; 30 drawn with seed 20261109) | 30 | 7.6e-10 (largest) | 7.6e-10 | 2.9e-9 |
| the 17 that moved before, `sparse_followers(4000, 10, 3)`, `long_horizon(2500, 10, 21)`, `long_horizon(300, 6, 17)`, `long_horizon(600, 3, 5)` | 21 | off by up to 1 | 16 within 1.8e-9, 5 off by 5e-5 to 0.018 | the same 5 |
| re-solved after (every data set where p2a re-solves at 1e-8) | 29 | 17 off by more than 1e-6, up to 0.68 | 1.5e-9 (largest) | 1.5e-9 |

The regular set's 2.9e-9 (p2a at 1e-14) is the oracle's own slack (ltmle fits at glm epsilon 1e-8, and its re-solve
stops at a score of 1e-4); the case pins use 1e-6. Eight of the 30 regular data sets have
positivity warnings in ltmle and in p2a, and on none of them does either estimator apply a rule.
The five that still disagree are the quasi-separated ones above. On three of them ltmle re-solves
its last or third node, because its glm fluctuation (with the weights scaled by their root mean
square, and glm.fit's deviance rule) leaves the score above 1e-3 where p2a's scalar solver does
not, and ltmle's re-solve bounds the offsets to `logit([1e-8, 1 - 1e-8])`, which moves the
prediction of the subjects beyond that bound; on the other two a regime's mean is within 0.02 of 0
and the two fluctuations stop at different points of a nearly flat score. The 29 re-solved data sets were wrong before at every tolerance (the
sweep measured only the tolerance dependence and did not show them); ltmle's glm fluctuation solves
those scores without a re-solve.

**Warnings** (in `LtmleResult::warnings` and the `saturation` record; the Display prints a
"Saturation" block only when a check fires, so unflagged output is unchanged):
- positivity: more than 1% of the subjects, pooled over the estimated regimes, with an unbounded
  cumulative probability of following the regime at the last node below the lower g bound
  (`ltmle:::CheckForVarianceWarning`; one line per regime with its share). On
  `long_horizon(300, 6, 17)` the counts are 10 and 137, as ltmle's;
- rare outcome: a 0/1 outcome with a mean below 0.03 (the same ltmle check);
- each constant Q node, with its range and constant (ltmle's no-variation rule);
- each re-solved targeting step, with the score it left and the offset bound;
- Q*_1 within 1e-8 of 0 or 1 for more than 1% of the subjects (logistic Q model only: a linear
  Q* is on the outcome's own scale); propensities numerically 0 or 1 (glm.fit's `10 * eps`
  threshold), named by the treatment columns.

**Default tolerance.** It stays 1e-8, ltmle's glm epsilon. With the rules, 1e-10 changes no count
in the table above beyond the quasi-separated five and does not remove them, and costs up to 1.15x
on the T = 10 rows ("Saturating regimes: before and after").

## 4. Regimes and contrasts

- `Regime::Static(v)`: one value for every node (`Regime::always()`, `Regime::never()`) or one
  value per node (`Static(vec![1, 0])`).
- `Regime::Dynamic(rules)`: exactly one `DynamicRule` per node,
  `d_t = above if L > threshold else 1 - above`, on a covariate observed at or before the node
  (`Regime::threshold_rules(columns, c)` gives `A_t = 1{L_t > c}`; ltmle's `rule`).
- `LtmleConfig.contrast`: the reference regime (default never treat). The result's `ate`, `se`,
  `z_stat`, `p_value` and CI are `psi(intervention) - psi(contrast)` with the differenced EIC;
  `None` estimates the single mean and leaves the contrast fields NaN.
- `RegimeKind` and `LtmleQModel` parse option strings (`"always"`, `"never"`, `"dynamic"`;
  `"logistic"`, `"linear"` and aliases).

## 5. Continuous outcomes

With the logistic Q model an outcome outside [0, 1] is mapped to [0, 1] by its range before the
fits, and psi, the EIC and `targeted_predictions` are scaled back (ltmle's behaviour with
`Yrange = NULL`); the result reports `outcome_range`. `LtmleQModel::Linear` is p2a's Gaussian
variant: least-squares Q fits (OLS's normal equations, the pivoted QR when ill-conditioned) and
`eps_t` the `H_t`-weighted mean residual among the followers, `Q*_t = Q_t(d) + eps_t`.

## 6. Inference

`SE = sd(EIC) / sqrt(n)` (ltmle `variance.method = "ic"`), CIs `psi -/+ qnorm(0.975) SE`, the
p-value `2 Phi(-|z|)`. ltmle's default `variance.method = "tmle"` is a substitution variance
estimator that needs extra targeted fits per node ("takes significantly more time", ltmle NEWS)
and is unavailable for non-binary outcomes; p2a keeps the influence-curve variance as a
documented convention, and the output says so.

## 7. R case

`validation/r/cases/causal_ltmle.R`, case `ltmle_two_period`, dataset
`validation/datasets/generated/ltmle_sim.csv` (600 rows; `L0, A1, L1, A2, Y` the two-period
binary design, `Yc` a continuous outcome, `R1, Lr, R2, Yr` a randomized design with a known
effect of 5.5, `L2, A3, Y3` the third time point). Key values:

| Quantity | Value |
|---|---|
| psi always / never (binary, two periods) | 0.5736132092515959 / 0.3255881665399437 |
| ATE (SE), z, p | 0.24802504271165215 (0.05306179615051192), 4.674267753924484, 2.950042547809292e-06 |
| psi dynamic `A_t = 1{L_{t-1} > 0}` (SE) | 0.45392016719826284 (0.032384516421308296) |
| psi (1, 0) | 0.5956272044613323 |
| gbounds (0.1, 0.5): psi never | 0.32392409968247965 (per-factor bounds would give 0.32635506045819096) |
| rescaled Yc: ATE (SE) | 2.808160267792386 (0.13487370329062567) |
| linear Yc: ATE (SE) | 2.804806711524769 (0.13571364292968557) |
| randomized: ATE (SE), truth 5.5 | 5.647189491841183 (0.14222217207100085) |
| three periods: ATE (SE) | 0.3953563630362491 (0.07074698162059881) |

Case `ltmle_saturation`, dataset `validation/datasets/generated/ltmle_saturation.csv` (2,500
rows; `L1a, L1b, A1, ..., L8a, L8b, A8, Y`): an R port of the `sparse_followers` fixture (two
uniform covariates per node, `A_t ~ plogis(0.3 L_t1 + 0.2 A_{t-1})`, a Bernoulli outcome in the
treatments), seed 3, the first seed in 1..200 whose never regime has a no-variation node in
ltmle, whose written-out never mean moves by more than 1e-3 between glm epsilon 1e-8 and 1e-14,
and where ltmle succeeds. Its reference is `ltmle::ltmle` 1.3-0 itself (seeded before each call,
as its re-solve restarts from `rnorm`):

| Quantity | Value |
|---|---|
| followers at the last node, always / never | 18 / 9 |
| psi always (SE) | 0.71327639856105862 (0.073041017618869344) |
| psi never (SE) | 4.0059606610570884e-11 (4.8057881059090657e-12) |
| ATE (SE), two-regime call | 0.71327639536478815 (0.073041017618861018) |
| no-variation Q nodes, never | A1 to A7 (none for always); no re-solve |
| positivity warning; subjects below the g bound, always / never | yes; 2,221 / 2,500 |
| written-out never mean at epsilon 1e-8 / 1e-14 (EIC mean at 1e-14) | 7.35e-13 / 0.0169 (-0.040) |
| analytic estimand, always / never (context only) | 0.82463 / 0.15605 |

## 8. Tests and tolerances

The registry entry is `validation/methods/causal_ltmle.json` (method `causal_ltmle`). At
n = 600: coefficients, means, epsilons, H and cumulative g 1e-8 relative; SEs, z and CI bounds
1e-6; p-values 1e-6 relative / 1e-12 absolute; counts exact. p2a runs at `tolerance = 1e-14`,
the case's `glm.control(epsilon = 1e-14)`; the default tolerance 1e-8 is checked against the
same tight values at `Tol::iterative(1e-8)` (measured gaps: 2.8e-13, 1.1e-11, 6.0e-15 and
4.1e-10 for psi always, psi never, the ATE SE and the never epsilons).

`test_validate_ltmle_saturation_matches_ltmle` (registry entry `causal_ltmle_saturation`, case
`ltmle_saturation`) runs the default tolerance against ltmle itself: the always mean, its SE, the
ATE and its SE at 1e-6 relative, the never mean and its SE (about 4e-11 and 5e-12) at 1e-8
absolute; the no-variation nodes of both regimes, the positivity and rare-events flags, the
followers and the counts below the g bound exact. The oracle's own slack, measured on 30 regular
data sets, is 2.9e-9 on the means and the ATE and 5.6e-10 on the SEs (ltmle fits at glm epsilon
1e-8). Measured gaps on the case: 1.8e-13 on the always mean, 3.2e-9 on the ATE (against ltmle's
two-regime call, whose ATE differs from the difference of its single-regime means by as much),
4.0e-11 and 4.8e-12 absolute on the never mean and SE, below 1e-14 on the other SEs.

## 9. Planning checks (plan T3-WP27)

- Saturated models (n = 2000, discrete L0 and L1, fully interacted g and Q): the TMLE equals the
  hand-computed g-formula to 1.1e-16 for (1, 1), (0, 0) and (1, 0); every epsilon is below 4e-15.
- Monte Carlo (400 replications of the case DGP at n = 600): ATE bias -0.0057 (truth 0.34616),
  mean EIC SE 0.0507 against the Monte Carlo SD 0.0489, 95% coverage 0.948 (always 0.958, never
  0.955, dynamic 0.970).
- Nonparametric bootstrap on the case data (B = 400): ATE SE 0.05278 against the EIC SE 0.05306.

## 10. Deviations from design A.24

1. `Q_t` is predicted with every `A_s, s <= t`, set to the regime (ltmle's `SetA`); A.24 set
   only `A_t`. The design values move (psi always 0.5736124939592598 -> 0.5736132092515959,
   ATE SE 0.0530591757778956 -> 0.05306179615051192).
2. `gbounds` bound the cumulative probability (ltmle's `CalcCumG` + `Bound`), default
   `(0.01, 1)`; A.24 bounded each factor at 0.01. The default bounds bind for no follower on
   the case data; the case pins `(0.1, 0.5)`, which binds.
3. The case keeps its id but also pins a continuous outcome, a randomized design with a known
   effect, three time points and the mixed static regimes (1, 0) and (1, 0, 1).

## MCP extraction (T2-WP11)

The MCP tool `ltmle` reads every node (the outcomes, the treatments and each time point's
comma-separated covariates) in one request to the shared extractor with the `Fail` policy. p2a's
LTMLE has no censoring node, and dropping a row would change the sample of every time point, so
a missing (null or NaN) value is a tool error naming the column. Integer and boolean columns are
numeric; no data problem raises a protocol error. Before T2-WP11 outcomes and treatments were read
with `into_no_null_iter`, so a missing final outcome counted as Y = 0 and a missing treatment as
untreated, with the full row count reported, and missing covariates became NaN and failed as a
singular matrix (finding TOOL-7, the handler part of CAUS-24).

Tests (`crates/p2a-mcp/src/tools/handlers/causal.rs`, on `rd_sim.csv` with arbitrary node roles;
not registered): `test_mcp_ltmle_accepts_int64_treatment` and `test_mcp_ltmle_refuses_missing_values`.

## 12. Limitations

- No censoring (C) nodes: missing values are refused.
- A single outcome node at the last time point; non-zero earlier `outcomes` entries are ignored
  with a warning (no intermediate or survival outcomes).
- No marginal structural models (`ltmleMSM`).
- No SuperLearner: main-term logistic (or linear) models only, no `Qform`/`gform` override.
- No `stratify = TRUE`.
- No `variance.method = "tmle"` (section 6).
- The no-variation rule applies to Q regressions only. ltmle applies it to g as well, but in p2a a
  constant treatment at a node is either refused (no follower of a regime) or followed by every
  subject, so a g fit never sees a constant response that matters.
- A last-node Q regression that is quasi-separated on a 0/1 outcome (30 or more IRLS steps) still
  gives a regime prediction that depends on where the deviance rule stops; on five of the sweep's
  1,715 data sets the ATE moves by 3.7e-6 to 0.023 between tolerances 1e-8 and 1e-14, and p2a and
  ltmle differ there by 5e-5 to 0.018 ("Saturating regimes").
- The full-history designs grow with the number of time points: node t has
  `1 + sum_{s <= t} k_s + t` columns, so the cost per regime grows as `T^3` (as in ltmle).
  See "Wall-clock against fe13b889" for the cost when a fluctuation saturates a regime.

## Wall-clock against fe13b889

Medians in ms over three interleaved rounds of the timing harness (`wp27`, one P core, one rayon
thread, `release`) at 811c26bb; ratio is this tree over fe13b889. fe13b889's estimator is not
comparable: it fitted shorter histories (its T = 2 growth ATE is -0.0029, this tree's 0.2397).

| Row | n | fe13b889 | this tree | ratio |
|---|---|---:|---:|---:|
| bench | 100 | 0.23 | 0.09 | 0.39 |
| bench | 1,000 | 1.78 | 0.73 | 0.41 |
| bench | 10,000 | 17.2 | 7.2 | 0.42 |
| linear Q | 100 | 0.07 | 0.05 | 0.72 |
| linear Q | 1,000 | 0.48 | 0.34 | 0.71 |
| linear Q | 10,000 | 4.7 | 3.5 | 0.74 |
| binary Y | 100 | 0.15 | 0.28 | 1.86 |
| binary Y | 1,000 | 1.23 | 0.81 | 0.66 |
| binary Y | 10,000 | 11.9 | 8.0 | 0.67 |
| growth T = 2 | 10,000 | 10.2 | 7.0 | 0.69 |
| growth T = 3 | 10,000 | 14.1 | 13.5 | 0.96 |
| growth T = 5 | 10,000 | 23.8 | 23.1 | 0.97 |
| growth T = 10 | 10,000 | 46.3 | 251.0 | 5.42 |

Binary Y at n = 100 costs 1.86x because one Q fit takes 33 steps at that sample size, the contrast
regime's first (Q and g steps 4, 4, 6, 5, 33, 5). This is accepted.

**Growth T = 10 costs about 5.5x on a saturating regime and about 1.46x otherwise; both are
accepted.** (Measured at 811c26bb, before ltmle's no-variation rule; the saturating rows have
since dropped to about a fifth, see "Saturating regimes: before and after" below.) The row's dataset has about 20 followers of each regime at the last node, and the
always-treat fluctuation there reaches `eps = 29.2`. Every earlier Q response of that regime is then
within about `exp(-29)` of 1, and each of those nine fits takes 30 IRLS steps in `fit_glm` as its
linear predictor climbs to the clamp, where the other regime's take 2. Of six data seeds at
T = 10, n = 10,000, two saturate this way:

| Data seed | fe13b889 | this tree | ratio |
|---|---:|---:|---:|
| 42, saturates | 45.4 | 251.1 | 5.53 |
| 3, saturates | 45.6 | 251.0 | 5.50 |
| 1 | 45.3 | 65.9 | 1.45 |
| 2 | 45.6 | 66.8 | 1.46 |
| 5 | 45.3 | 66.4 | 1.47 |
| 7 | 45.5 | 65.9 | 1.45 |

Those fits keep glm.fit's start. Starting `fit_glm` from `warm_start` on them is not done: at
f89d777e it took seed 42 from 437 ms to 122 ms and seed 3 from 445 ms to 110 ms, with the ATE, SE,
regime means, EIC, Q* and fluctuation coefficients within 5e-11 and equal warnings and convergence
status, but the reported Q coefficients of the saturated nodes moved by up to 2.7, because their
likelihood is flat.

A fit with a row past the `|eta| = 15` bound pays for the lean fit and then `fit_glm`, and reports
`fit_glm`'s numbers and warnings. No fit in the rows above leaves that way. The bound's cost on such
designs is measured on IPW (`treatment_ipw.md`) and C-TMLE (`ctmle.md`).

### Saturating regimes: before and after

Medians in ms over five interleaved rounds of the same harness (the rows above plus the three
saturating fixtures), one core, one rayon thread, `release`, load average 2.5 to 3.4. "Before" is
the code without the no-variation rule and the score re-solve, "after" the code with them, both
at the default tolerance of 1e-8; the last column is the cost of a tolerance of 1e-10 instead.

| Row | n | before | after | after/before | 1e-10 / 1e-8 |
|---|---|---:|---:|---:|---:|
| bench | 100 / 1,000 / 10,000 | 0.092 / 0.748 / 7.31 | 0.090 / 0.739 / 7.41 | 0.98 / 0.99 / 1.01 | 1.02 / 1.04 / 1.07 |
| linear Q | 100 / 1,000 / 10,000 | 0.050 / 0.365 / 3.67 | 0.049 / 0.375 / 3.67 | 0.98 / 1.03 / 1.00 | 1.00 / 0.99 / 1.01 |
| binary Y | 100 | 0.285 | 0.105 | 0.37 | 1.07 |
| binary Y | 1,000 / 10,000 | 0.831 / 8.09 | 0.830 / 8.14 | 1.00 / 1.01 | 1.01 / 1.03 |
| growth T = 2 / 3 / 5 | 10,000 | 7.12 / 13.6 / 23.4 | 7.14 / 13.6 / 23.7 | 1.00 / 1.00 / 1.01 | 1.02 / 1.02 / 1.02 |
| growth T = 10 (seed 42, saturates) | 10,000 | 260.8 | 56.0 | 0.21 | 1.08 |
| T = 10, seed 3 (saturates) | 10,000 | 264.7 | 57.7 | 0.22 | 1.09 |
| T = 10, seeds 1 / 2 / 5 / 7 | 10,000 | 67.2 / 68.4 / 68.0 / 67.5 | 67.5 / 68.6 / 68.0 / 68.3 | 1.00 / 1.00 / 1.00 / 1.01 | 1.09 / 1.07 / 1.13 / 1.15 |
| `shifted_outcome((5000, 10, 4), 3, false, 2)` | 5,000 | 528.5 | 24.4 | 0.05 | 1.07 |
| `sparse_followers(2500, 8, 20)` | 2,500 | 38.4 | 9.9 | 0.26 | 1.07 |
| `sparse_followers(4000, 10, 3)` | 4,000 | 101.0 | 28.2 | 0.28 | 1.05 |

At the default tolerance no row is slower, and the saturating rows lose the IRLS runs of their
constant nodes (the ATEs of the T = 10 saturating seeds are unchanged to six digits). Binary Y at
n = 100 drops to 0.37x because its 33-step Q fit had a response without variation. A tolerance of
1e-10 costs 1.02x to 1.15x, most on the T = 10 rows that do not saturate.

## 13. References

- van der Laan, M.J. & Gruber, S. (2012). "Targeted Minimum Loss Based Estimation of
  Causal Effects of Multiple Time Point Interventions." *The International Journal
  of Biostatistics*, 8(1), Article 9. https://doi.org/10.1515/1557-4679.1370

- Lendle, S.D., Schwab, J., Petersen, M.L., & van der Laan, M.J. (2017). "ltmle:
  An R Package Implementing Targeted Minimum Loss-Based Estimation for Longitudinal
  Data." *Journal of Statistical Software*, 81(1), 1-21. https://doi.org/10.18637/jss.v081.i01

- van der Laan, M.J. & Rose, S. (2011). *Targeted Learning: Causal Inference for
  Observational and Experimental Data*. Springer. https://doi.org/10.1007/978-1-4419-9782-1

- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and
  Causal Inference Models." *Biometrics*, 61(4), 962-973.

- R package `ltmle`: https://cran.r-project.org/package=ltmle (1.3-0 for the cross-check).

## 14. Rust Test Functions

R-pinned (registered under `causal_ltmle`; case `ltmle_two_period`):
`test_validate_ltmle_ate_se`, `test_validate_ltmle_default_control`, `test_validate_ltmle_linear_q`,
`test_validate_ltmle_randomized_known_effect`, `test_validate_ltmle_logistic_rescaled_outcome`,
`test_validate_ltmle_gbounds_cumulative`, `test_validate_ltmle_three_period`,
`test_validate_ltmle_intervention_changes_estimate`, `test_validate_ltmle_always`,
`test_validate_ltmle_never`, `test_validate_ltmle_dynamic`, `test_validate_ltmle_static_mixed`,
`test_validate_ltmle_three_period_regimes`, `test_validate_ltmle_nuisance_models`,
`test_validate_ltmle_duplicated_history_column_is_aliased`. Case `ltmle_saturation` (entry
`causal_ltmle_saturation`, reference `ltmle::ltmle` itself): `test_validate_ltmle_saturation_matches_ltmle`.

Property and error tests (on `ltmle_sim.csv` without loading the case, or on inline data; not
registered): `test_ltmle_config_validation_errors`, `test_ltmle_no_follower_is_an_error`,
`test_ltmle_too_few_observations`, `test_ltmle_data_rejects_non_finite`,
`test_ltmle_regime_validation_errors`, `test_regime_kind_parses_aliases_and_rejects_unknown`,
`test_ltmle_q_model_parses_aliases_and_rejects_unknown`,
`test_ltmle_display_reports_regimes_and_notes`, `test_ltmle_covariate_scale_invariance`,
`test_ltmle_intermediate_outcomes_warn`, `test_ltmle_data_validation`,
`test_ltmle_data_validation_errors`, `test_intervention_type_display`,
`test_ltmle_single_time_point_error`, `test_ltmle_earlier_q_fits_start_from_the_later_node`,
`test_ltmle_fast_paths_match_the_shared_glm`, `test_ltmle_fast_paths_match_on_other_regimes`,
`test_ltmle_fast_paths_leave_for_collinear_and_separated_nodes`,
`test_ltmle_fast_paths_keep_the_outcome_at_a_low_max_iter`,
`test_ltmle_fast_paths_report_a_saturated_regime_through_the_shared_glm`,
`test_ltmle_fast_paths_hold_the_gaps_measured_off_the_fixtures`,
`test_ltmle_linear_q_matches_after_the_row_predictions`, `test_ltmle_saturation_is_reported_and_warned`,
`test_ltmle_no_saturation_on_regular_data`,
`test_ltmle_linear_q_outcome_off_unit_interval_is_not_saturation`,
`test_ltmle_propensity_at_bound_warning_names_the_treatment`, `test_ltmle_rare_outcome_is_warned`,
`test_ltmle_constant_q_response_is_not_regressed`, `test_ltmle_constant_outcome_is_refused`,
`test_ltmle_saturating_fixtures_agree_with_ltmle` (values quoted from ltmle 1.3-0 on the exported
fixtures), `test_ltmle_targeting_solves_the_score_at_every_node`,
`test_resolve_fluctuation_monotone_root`.

## 15. Status

R-pinned: the logistic-Q quantities to the written-out ltmle algorithm, cross-checked against
ltmle 1.3-0; the linear-Q quantities to a written-out reference; a saturating regime
(`ltmle_saturation`) to ltmle 1.3-0 itself.
