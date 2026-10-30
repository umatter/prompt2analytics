# Regression Discontinuity (rdrobust): sharp RD with CCT bias correction

**Status:** R-pinned at fixed bandwidths (registry `validation/methods/causal_rd.json`, id
`rd_sharp`), with one known deviation: the bandwidths p2a selects when none are given are
its own (finding CAUS-28).

## Method and reference

| Package | Function | Version tested |
|---|---|---|
| rdrobust | `rdrobust(y, x, c, p, q, h, b, kernel, vce, nnmatch, level)` | 4.0.0 (R 4.3.3) |

Calonico, Cattaneo & Titiunik (2014), "Robust Nonparametric Confidence Intervals for
Regression-Discontinuity Designs", *Econometrica* 82(6), 2295-2326.

p2a functions: `run_rd` (sharp), `rd_bandwidth` (bandwidths only), and the per-cutoff fits of
`run_rd_multi` (`validation/econometrics/rdmulti.md`).

## What p2a computes

For given bandwidths, `run_rd` computes what `rdrobust()` 4.0.0 computes. Bandwidths are given
as `RdConfig.h` and `RdConfig.b`, and `h_right`/`b_right` when the right side differs (rdrobust
`h = c(h_l, h_r)`). Without `b`, `b = rho * h` with `rho = 1`, as rdrobust does when only `h`
is given. The order is `p` (default 1), and the bias-correction order `q` (default `p + 1`)
must exceed `p`, as rdrobust requires.

Per side s (left: x < c; right: x >= c), with bandwidths (h, b):

1. Kernel weights `W_h = K((x - c)/h)/h` and `W_b = K((x - c)/b)/b`, with K triangular
   `1 - |u|`, Epanechnikov `0.75 (1 - u²)` or uniform `0.5` on `|u| <= 1` (rdrobust's
   `rdrobust_kweight`; `|u| = 1` has weight 0 except for the uniform kernel).
2. The rows that enter the fits are those with positive weight under the larger of h and b
   (rdrobust's `ind`), sorted by x (ties keep their row order). `N_h` and `N_b` count the rows
   with positive weight under h and under b.
3. `R_q = [1, (x - c), ..., (x - c)^q]`, `R_p` its first p + 1 columns, `u = (x - c)/h`,
   `invG_p = (R_p' W_h R_p)^-1`, `invG_q = (R_q' W_b R_q)^-1`, `L = R_p' (W_h u^(p+1))`.
4. The bias-correction weights (rdrobust's `Q_q`):
   `Q_i = W_h,i R_p,i - h^(p+1) W_b,i (R_q,i · invG_q[:, p+1]) L`.
5. `beta_p = invG_p R_p' W_h y`, `beta_q = invG_q R_q' W_b y`, `beta_bc = invG_p Q' y`.
6. Residuals.
   - `vce = "nn"` (default): rows with equal x form a block; each row's neighbours are its
     own block plus whole neighbouring blocks, added on the nearer side (both when equally
     far), until there are at least `min(nnmatch, m - 1)` others (m rows in the window). With
     `J_i` neighbours, `res_i = sqrt(J_i / (J_i + 1)) (y_i - mean of the neighbours)`
     (rdrobust's `rdrobust_res`). The same residuals serve both variances.
   - `vce = "hc0" .. "hc3"`: `res_h = w_i (y_i - R_p,i beta_p)` and
     `res_b = w_i (y_i - R_q,i beta_q)`, with `w = 1` (HC0), `sqrt(m / (m - d))` (HC1,
     `d = p + 1` or `q + 1`, m the rows in the window), `sqrt(1 / max(1 - h_ii, 1e-8))` (HC2)
     or `1 / max(1 - h_ii, 1e-8)` (HC3), with the leverage `h_ii = W_i R_i' invG R_i` of the
     order-p fit for `res_h` and of the order-q fit for `res_b`.
7. Variances of the intercept, with `a` the first row of `invG_p`:
   `V_cl = sum_i res_h,i² (a · W_h,i R_p,i)²` and `V_rb = sum_i res_b,i² (a · Q_i)²`
   (rdrobust's `invG_p rdrobust_vce(..) invG_p` at [1, 1]).

Then `tau_cl = beta_p,right[0] - beta_p,left[0]`, `tau_bc = beta_bc,right[0] - beta_bc,left[0]`,
`se_cl = sqrt(V_cl,left + V_cl,right)`, `se_rb = sqrt(V_rb,left + V_rb,right)` and
`z = Φ⁻¹(1 - (1 - level)/2)`. The three rows of the result are rdrobust's:

| Row | Estimate | SE | Interval |
|---|---|---|---|
| Conventional | `tau_cl` | `se_cl` | `tau_cl ∓ z se_cl` |
| Bias-Corrected | `tau_bc` | `se_cl` | `tau_bc ∓ z se_cl` |
| Robust | `tau_bc` | `se_rb` | `tau_bc ∓ z se_rb` |

p-values are `2 Φ(-|est / se|)` (upper tail). `bias = tau_cl - tau_bc`. `n_eff_left/right`
are `N_h`, `n_b_left/right` are `N_b`, and `coef_left/right` are `beta_p` on each side.
`significance` follows the robust p-value.

The cost per side is O(m log m) for the sort, O(m (q + 1)²) for the fits and O(m + B · nnmatch)
for the nearest-neighbour residuals (B distinct values of x).

## Reference cases

`validation/r/cases/causal_rd.R` simulates `validation/datasets/generated/rd_sim.csv` (seed 42,
n = 1000, cutoff 0): `y` a sharp design with a jump of 0.4, `d` an integer treatment whose
take-up jumps from 0.2 to 0.8, `yf` an outcome that responds to `d`, and `xm`/`ym` a running
variable on a 0.01 grid (201 distinct values) with its outcome.

| Case | Calls |
|---|---|
| `rd_sharp_fixed_bw` | `rdrobust(y, x, c = 0, p = 1, h = 0.3, b = 0.3, kernel = k, vce = v)` for three kernels × five VCEs (`<k3>_<vce>_hb`); h 0.3 b 0.5 and h 0.5 b 0.3 (nn); p = 2 with h 0.5 b 0.6; `h = c(0.25, 0.35), b = c(0.4, 0.45), vce = "hc1"` (`tri_hc1_asym`); `level = 90` |
| `rd_mass_points_nn` | `rdrobust(ym, xm, c = 0, h = 0.3, b = 0.4, vce = "nn", nnmatch = m)`, m = 3, 1, 5 |

Each run records `coef_conv`, `coef_bc`, `se_conv`, `se_robust`, `z_*`, `p_*`, `ci_conv`,
`ci_robust`, `n_h`, `n_b` and the order-p coefficients on each side.

## Decisive values

R 4.3.3, rdrobust 4.0.0; full precision in `validation/expected/*.json`.

| prefix | coef_conv | coef_bc | se_conv | se_robust | ci_robust | p_robust | N_h / N_b |
|---|---|---|---|---|---|---|---|
| `tri_nn_hb` | 0.5817841314098199 | 0.5919701612168791 | 0.12379231806549339 | 0.19836524289095045 | 0.2031814293660763, 0.9807588930676818 | 0.0028428055888784203 | [161, 138] / [161, 138] |
| `tri_hc0_hb` | 0.5817841314098199 | 0.5919701612168791 | 0.11983035927521032 | 0.18417006087364127 | 0.23100347487399292, 0.9529368475597653 | 0.0013078193187529127 | same |
| `tri_hc1_hb` | 0.5817841314098199 | 0.5919701612168791 | 0.12065335399202394 | 0.18611035248894434 | 0.22720057318849385, 0.9567397492452643 | 0.001468952669997585 | same |
| `tri_hc2_hb` | 0.5817841314098199 | 0.5919701612168791 | 0.12120484202992325 | 0.18863159916515235 | 0.22225902050698482, 0.9616813019267734 | 0.0016996893310610322 | same |
| `tri_hc3_hb` | 0.5817841314098199 | 0.5919701612168791 | 0.12260010167716745 | 0.19326435944084766 | 0.21317897721761414, 0.970761345216144 | 0.0021912464836270316 | same |
| `epa_nn_hb` | 0.5805845466472614 | 0.6249259972600473 | 0.1193314083641351 | 0.189339186769562 | 0.25382801032960317, 0.9960239841904914 | 0.0009649089662596728 | same |
| `epa_hc1_hb` | 0.5805845466472614 | 0.6249259972600473 | 0.11756630567553424 | 0.17984579111190654 | 0.2724347239095968, 0.9774172706104978 | 0.0005112572215066532 | same |
| `uni_nn_hb` | 0.5290400723570072 | 0.6728127020375516 | 0.1118504516281861 | 0.17725367422399307 | 0.32540188443112955, 1.0202235196439737 | 0.00014719085271196658 | same |
| `uni_hc3_hb` | 0.5290400723570072 | 0.6728127020375516 | 0.1123653908338009 | 0.1743305189659708 | 0.3311311634580721, 1.0144942406170312 | 0.00011366176323095737 | same |
| `tri_nn_h03_b05` | 0.5817841314098199 | 0.6137938269272958 | 0.12379231537314167 | 0.14640367896736542 | 0.32684788894709543, 0.9007397649074962 | 2.759269746236069e-05 | [161, 138] / [259, 227] |
| `tri_nn_h05_b03` | 0.50788517130959 | 0.533453843207355 | 0.09450716219111888 | 0.3350712798446803 | -0.12327379754196, 1.19018148395667 | 0.11137104837279291 | [259, 227] / [161, 138] |
| `tri_nn_p2` | 0.5911960428589791 | 0.6399605098112159 | 0.1423006600416738 | 0.17491812752108568 | 0.2971272796267036, 0.9827937399957282 | 0.0002535682356256111 | [259, 227] / [308, 277] |
| `tri_hc1_asym` | 0.5680738086557416 | 0.6301640237245245 | 0.11791412807110979 | 0.14761279255341475 | 0.3408482666624494, 0.9194797807865996 | 1.9632138393300462e-05 | [139, 157] / [216, 204] |
| `nn3` | 0.19249811566249087 | 0.16395388908486985 | 0.1302920671764542 | 0.16371764508097778 | -0.15692679890755767, 0.4848345770772974 | 0.3166126858419984 | [153, 140] / [211, 183] |
| `nn1` | same | same | 0.12994764004208972 | 0.1633561567021435 | -0.15621829470421272, 0.48412607287395243 | 0.3155429696397453 | same |
| `nn5` | same | same | 0.12966001128452542 | 0.162979271926044 | -0.15547961411673622, 0.4833873922864759 | 0.31442518763179694 | same |

`level90_ci_robust` = [0.2656883719865894, 0.9182519504471687]. `tri_nn_h03_b05_se_conv`
differs from `tri_nn_hb_se_conv` in the 9th digit: the nearest neighbours are searched in the
larger (b) window. The `epa`/`uni` HC0 and HC2 rows are in the JSON.

Before T2-WP11 p2a's NN variance was the kernel-weighted mean of a per-observation σ² divided
by the count in the window (se_conv about 0.0697 on `tri_nn_hb`, half of rdrobust's), the bias
was replaced by 10% of the estimate whenever it exceeded half of it (tau_bc 0.5048), and the
robust SE was the conventional one times a constant (findings CAUS-1, CAUS-5, CAUS-29).

## Tests and tolerances

Registered (`rd_sharp`):

- `test_validate_rd_sharp_fixed_bw` (p2a-core, `econometrics::rd::tests`): every run of
  `rd_sharp_fixed_bw`, every recorded quantity.
- `test_validate_rd_mass_points_nn`: `rd_mass_points_nn`, nnmatch 3, 1 and 5.
- `test_validate_rd_sharp_display_table`: the printed table rows (`tri_hc1_hb`), the `[95% CI]`
  and `[90% CI]` headers and the bias-window counts.

Not registered (no R value): `test_nn_residuals_follow_rdrobust_rule` holds the block-based
nearest-neighbour residuals to a line-by-line transcription of rdrobust's `rdrobust_res` loop
on continuous, tied, signed-zero and tiny inputs, and
`test_rd_rejects_q_not_above_p_and_lonely_right_bandwidths` checks the `q > p` rule and the
right-side bandwidth checks.

Tolerances (`validation/README.md`, n = 1000): coefficients 1e-8 relative, SEs, statistics
and CI bounds 1e-6 relative, p-values 1e-6 relative plus 1e-12 absolute, counts exact. The
observed agreement with rdrobust is at most about 1.7e-12 relative.

## Missing values and column types

`run_rd`, `run_fuzzy_rd` and `rd_bandwidth` read their columns through the shared extractor
(`p2a_core::data::columns`) with the `Omit` policy. A row with a missing outcome, running
variable or (fuzzy) treatment is dropped from every column, as rdrobust does
(`na.ok <- complete.cases(x, y)`, plus the `fuzzy` column). Null and NaN both count as missing.
The result reports the drop in `missing` (`n_total`, `n_used`, `n_dropped`, `by_column`), and
the printed output ends with one line such as
`Missing values: 4 of 1000 rows dropped because of missing values (y: 2, x: 2)`. Infinite
values are refused with `NonFiniteValues` and text columns with `NonNumericColumn`, both
naming the column; an absent column is `ColumnNotFound`. Integer and boolean columns are
numeric. Before T2-WP11 every extraction problem was reported as "column not found", and a NaN
surfaced as a singular matrix (finding CAUS-76).

The case `rd_sharp_fixed_bw` records an rdrobust run on a copy of `rd_sim` with two `y` cells
and two `x` cells inside h = 0.3 set to NA (0-based rows `na_rows_y` = [10, 32] and
`na_rows_x` = [73, 123]): `na_tri_nn_hb` (996 rows; coef_conv 0.5689702543258852, coef_bc
0.5714073393211437, se_conv 0.1248633251364192, se_robust 0.19907535854441885, ci_robust
[0.18122680636468474, 0.9615878722776028], p_robust 0.004100738107604092, N_h [158, 137]).

- `test_validate_rd_sharp_fixed_bw_missing_rows` (registered) nulls, and in a second pass sets
  to NaN, the same cells of the committed CSV and pins every `na_tri_nn_hb` quantity (n = 996
  tolerances), the `MissingSummary` and the last output line.
- `test_na_rd_refuses_infinite_text_and_absent_columns` (not registered) checks the three
  refusals through `run_rd`, `rd_bandwidth` and `run_fuzzy_rd`.
- `test_na_rd_bandwidth_and_fuzzy_drop_missing_rows` (not registered) checks that
  `rd_bandwidth` with missing outcomes and `run_fuzzy_rd` with missing treatments equal the
  runs on the filtered rows and report what was dropped.

## Fuzzy RD

**Status:** R-pinned at fixed bandwidths (registry id `rd_fuzzy`, deviation CAUS-28 for the
bandwidths p2a selects itself).

`run_fuzzy_rd(dataset, y, x, t, c, config)` computes what `rdrobust(y, x, c, fuzzy = t)`
computes. The bandwidths are resolved once (when `h` is not given p2a's selector runs on the
outcome), and the outcome and the treatment are fitted together: `D = [y, t]` goes through the
steps above with the same matrices, so both stages share every bandwidth and window. With the
conventional jumps `tau_Y`, `tau_T` and the bias-corrected ones `tau_Y,bc`, `tau_T,bc`:

- conventional estimate `tau = tau_Y / tau_T` (rdrobust `coef[1]`);
- `s = (1/tau_T, -tau_Y/tau_T²)` and the bias-corrected estimate
  `tau_bc = tau - s · (tau_Y - tau_Y,bc, tau_T - tau_T,bc)` (rdrobust `coef[2]`);
- the variances use the combined residuals `r_i = res_i · s = res_Y,i/tau_T - tau_Y/tau_T² res_T,i`
  in step 7, so the covariance of the two jumps is included: `se_fuzzy` (rdrobust `se[1]`) and
  `se_fuzzy_robust` (`se[3]`);
- intervals `tau ∓ z se_fuzzy` and `tau_bc ∓ z se_fuzzy_robust` at `config.level`;
- the first stage `first_stage_tau = tau_T` and `first_stage_f = (tau_T / se_T)²` with the
  conventional SE of the first stage (rdrobust `tau_T[1]`, `se_T[1]`); a warning when F < 10.

`outcome_rd` and `first_stage` are the sharp results of the two columns from the same fit:
rdrobust's `se_T` equals the SE of the sharp RD of `t` on the same bandwidths.
`significance` follows the robust p-value.

Before T2-WP11 the two stages were separate `run_rd` calls (two selector runs when `h` was not
given), the delta-method SE had no covariance term (0.2598 at h = b = 0.4 against rdrobust's
0.2413), and the interval was always 95% (finding CAUS-30).

Case `rd_fuzzy_fixed_bw` (same dataset): `rdrobust(yf, x, c = 0, fuzzy = d, p = 1, h = 0.4,
b = 0.4, vce = v)` for `nn` and `hc1`, `h = 0.3, b = 0.5` (nn), `level = 90`, the reduced form
`rdrobust(yf, x, h = 0.4, b = 0.4)`, and a weak first stage at `h = b = 0.2`.

| prefix | coef_conv | coef_bc | se_conv | se_robust | ci_robust | first stage conv / bc / se_conv / se_robust |
|---|---|---|---|---|---|---|
| `fuzzy_nn` | 0.6416930220442466 | 0.690755958013317 | 0.24134054288160955 | 0.3470342530233884 | 0.01058132068571549, 1.3709305953409185 | 0.4211742981234006 / 0.45274042362516853 / 0.10055911993020157 / 0.15050476117932227 |
| `fuzzy_hc1` | 0.6416930220442466 | 0.690755958013317 | 0.24372095030116525 | 0.3440482430829733 | 0.01643379262640765, 1.3650781234002265 | same / same / 0.09321659408857996 / 0.14087306353620874 |
| `fuzzy_nn_h03_b05` | 0.6379259283424029 | 0.6423291960699518 | 0.2604196887356571 | 0.30504621808028354 | 0.04444959501244505, 1.2402087971274585 | 0.4444905369329633 / 0.422598284293907 / 0.11664767556060084 / 0.13643660973802818 |

N_h/N_b: `fuzzy_nn` [216, 180] / [216, 180]; `fuzzy_nn_h03_b05` [161, 138] / [259, 227].
`reduced_form_coef_conv` 0.2702646081701694; `fuzzy_level90_ci_conv`
[0.24472315475499373, 1.0386628893334995] and `fuzzy_level90_ci_robust`
[0.11993540825140148, 1.2615765077752326]. First-stage F: 17.5420681500485 (`fuzzy_nn`),
14.520208773934074 (`fuzzy_nn_h03_b05`) and 9.898269468659171 at h = b = 0.2
(`fuzzy_nn_h02_first_stage_*`: 0.4572994934088912, SE 0.14535202408748185).

Tests (registered under `rd_fuzzy`):

- `test_validate_rd_fuzzy_fixed_bw`: every fuzzy run and quantity above, the reduced form and the
  90% intervals; both stages report the same bandwidths.
- `test_validate_rd_sharp_int64_outcome`: the sharp RD of the Int64 column `d` equals rdrobust's
  first stage (`fuzzy_nn_first_stage_*`).
- `test_validate_rd_fuzzy_weak_first_stage_warning`: at h = b = 0.2 the first stage and its F
  (9.90) match, and the result carries the warning `Weak first stage: F = 9.90 (< 10); ...`;
  at h = b = 0.4 it does not.

Not registered (no R value): `test_fuzzy_rd_one_bandwidth_for_both_stages` checks that the
selector runs once, on the outcome, and that both stages use its bandwidths, on a curved design
(n = 2000, x uniform on [-1, 1], take-up 0.2 / 0.8, `y = 0.5 + 10 x² + d + N(0, 0.3²)`) where
the selector run on each column alone picks h = 0.4241 for the outcome and 0.4996 (the cap, a
quarter of the range) for the treatment; the test first asserts that the two differ, so two
separate selector runs fail it.

## MCP and CLI

MCP tools (`crates/p2a-mcp/src/tools/handlers/causal.rs`): `rd_estimate`, `rd_fuzzy` and
`rd_multi` take `vce` (`"nn"`, the default, or `"hc0"` to `"hc3"`, case-insensitive; any other
value is a tool error listing them, checked before the dataset is read), and `rd_fuzzy` takes
the bias bandwidth `b` (default `h`). The text they return is the core `Display` output above.

- `test_validate_mcp_rd_estimate_vce` (registered under `rd_sharp`): the Conventional and
  Robust rows of `tri_nn_hb` (no `vce`, and `"nn"`), `tri_hc1_hb` (`"hc1"`) and `tri_hc3_hb`
  (`"HC3"`) are in the output, and `level = 0.9` prints `[90% CI]` with `level90_ci_robust`.
- `test_validate_mcp_rd_fuzzy_b` (registered under `rd_fuzzy`): `h = 0.3, b = 0.5` prints the
  `fuzzy_nn_h03_b05` rows and `F = 14.52`; `h = 0.4, vce = "hc1"` the `fuzzy_hc1` rows.
- `test_validate_mcp_rd_multi_vce` (registered under `rd_multi`): one cutoff, bandwidth 0.3,
  `vce = "hc1"` prints `tri_hc1_hb`'s bias-corrected estimate and robust SE.
- Not registered: `test_mcp_rd_estimate_rejects_unknown_vce` (`vce = "robust"` is a tool error
  for the three tools) and `test_mcp_rd_estimate_reports_dropped_rows` (a missing outcome is
  dropped and the `Missing values:` line printed).

CLI (`crates/p2a-cli/src/commands/causal.rs`): `p2a causal rd` and `p2a causal fuzzy-rd` take
`--h` (alias `--bandwidth`), `--b` (alias `--bias-bandwidth`) and `--vce` (default `nn`; an
unknown value is an error before the dataset is read). `causal rd` prints the robust estimate,
SE, p-value and CI, then the conventional estimate, SE and CI, the counts inside h and b, both
bandwidths and the VCE, and `-F json` adds the conventional and bias-corrected fields, the
counts, `b_left`/`b_right`, `kernel`, `vce`, `p`, `q`, `level`, `warnings` and `missing`.
`causal fuzzy-rd` prints the conventional LATE (as before, now labelled conventional), the
robust bias-corrected LATE with its SE, p-value and CI, and the first stage with its SE and F;
its JSON adds the robust fields, `first_stage_tau`, `first_stage_se`, `first_stage_f`,
`reduced_form_tau`, the bandwidths, counts, `vce`, `level`, `warnings` and `missing`. A drop of
rows with missing values ends the text output with a `Note:` line.

- `test_validate_cli_causal_rd_vce` (registered under `rd_sharp`): `-F json` at full tolerance
  and the printed lines at print precision, for `tri_nn_hb` (default `--vce`) and `tri_hc1_hb`.
- `test_validate_cli_causal_fuzzy_rd` (registered under `rd_fuzzy`): `--h 0.3 --b 0.5` against
  `fuzzy_nn_h03_b05`, first-stage F included.
- Not registered: `test_cli_causal_rd_rejects_unknown_vce` and
  `test_cli_causal_rd_missing_value_note` (one blank `y` field gives
  `Note: 1 of 1000 rows dropped because of missing values (y: 1)` and `missing.n_dropped = 1`).

## Limits

- Bandwidth selection is p2a's own MSE-type selector, not `rdbwselect` (CAUS-28, a registry
  deviation); every reference case passes h and b.
- No covariates, clusters, weights or derivatives (`deriv`).
- p2a's `rho` is b/h, rdrobust's is h/b; both default to b = h.
- `q > p` is required, as in rdrobust.
