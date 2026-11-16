# Validation: t-SNE

## Method

`tsne` (`crates/p2a-core/src/ml/reduction.rs`) is bhtsne's exact-gradient t-SNE with `Rtsne()`'s optimiser
defaults; the R reference is `Rtsne::Rtsne(theta = 0, pca = FALSE, normalize = FALSE)`.

1. **Affinities.** For each point, a search over the precision `beta = 1 / (2 sigma^2)` as bhtsne's
   `computeGaussianPerplexity`: start at 1; double (or halve) while the other bound is open, bisect once both
   are; at most 200 steps; stop when `|H - ln(perplexity)| < 1e-5`, with
   `H = beta * sum_j d2_ij p_j / S + ln S`, `p_j = exp(-beta d2_ij)` and `S = f64::MIN_POSITIVE + sum_j p_j`.
   Every row is written (with the last bandwidth tried when the search misses), and the misses are reported as
   one warning naming the number of points and the largest miss. `P` is `(P + P') / sum(P + P')`, with no floor.
2. **Optimiser.** Gradient `dY_i = sum_j (ex * P_ij - Q_ij / sum_Q) Q_ij (y_i - y_j)` with
   `Q_ij = 1 / (1 + |y_i - y_j|^2)` (no factor 4); `ex = 12` for iterations 0-249, then 1; gains +0.2 when the
   signs of `dY` and of the update differ, else x0.8, floor 0.01; update `uY = mom * uY - eta * gains * dY`,
   `mom = 0.5` for iterations 0-249, then 0.8; `Y += uY`, then `Y` is centred.
3. **Reported KL.** `sum_{i != j} P_ij ln((P_ij + FLT_MIN) / (q_ij + FLT_MIN))`, `q_ij = Q_ij / sum_Q`, of the
   returned `Y` with the unexaggerated `P` (Rtsne's cost formula).

### Defaults

| Setting | p2a `tsne` | `Rtsne()` default |
|---|---|---|
| start | N(0, 1e-4) per coordinate from `StdRng::seed_from_u64(seed.unwrap_or(42))` | N(0, 1e-4) from R's RNG |
| gradient | exact | Barnes-Hut, `theta = 0.5` (exact with `theta = 0`) |
| exaggeration | 12 for 250 iterations | `exaggeration_factor = 12`, `stop_lying_iter = 250` |
| momentum | 0.5, then 0.8 from iteration 250 | `momentum = 0.5`, `final_momentum = 0.8`, `mom_switch_iter = 250` |
| learning rate | `learning_rate` = `eta`, default 200 (sklearn's `learning_rate` is 4 x this) | `eta = 200` |
| iterations | 1000 | `max_iter = 1000` |
| perplexity | 30, must be below n / 3 | 30, `3 * perplexity <= n - 1` |
| preprocessing | none: the columns are used as given | `pca = TRUE` (to 50 dimensions), `normalize = TRUE` (centre, scale by the largest absolute value) |

Before ML-97 (Tier 4) the start was the PCA scores times 1e-4 whatever the seed, the exaggeration was 4 for 100
iterations and the gradient carried a factor 4 (an `eta` of 800 at `learning_rate = 200`). Before ML-37 the
bandwidth search bisected sigma linearly over [1e-10, 1e10] for 50 steps and wrote a row only when it converged:
212 of 300 rows were all zero on standard-normal data in 20 dimensions at perplexity 30, and `P` was floored at
1e-12.

## R code

`validation/r/cases/ml_reduction_t4.R`, case `tsne_rtsne_exact` (R 4.3.3, Rtsne 0.17):
`Rtsne(X, dims = 2, perplexity, theta = 0, pca = FALSE, normalize = FALSE, Y_init, max_iter = 1 | 10,
eta = 200, exaggeration_factor = 12, stop_lying_iter = 250, mom_switch_iter = 250, momentum = 0.5,
final_momentum = 0.8)`. The two switches are passed explicitly: with `Y_init` given, Rtsne sets both to 0.
Data: `generated/tsne_blobs.csv` (60 x 5, two shifted groups, perplexity 10) and `generated/tsne_unit.csv`
(300 x 20 standard normal, perplexity 30); each carries its start in `y0_1`, `y0_2` (N(0, 1e-4)).

## Values

| Key | R |
|---|---|
| `blobs_y_iter1`, `blobs_y_iter10` | the 60 x 2 embedding after 1 and 10 iterations, column-major |
| `unit_y_iter1` | the 300 x 2 embedding after 1 iteration, column-major |

Only 1 and 10 iterations are pinned: the path is chaotic in the summation order, and after 50 iterations two
correct implementations already differ by O(1).

## Tests and tolerances

| Test (`crates/p2a-core/src/ml/reduction.rs`) | Keys | Tolerance |
|---|---|---|
| `test_validate_tsne_rtsne_exact` | `blobs_y_iter1`, `unit_y_iter1`; `blobs_y_iter10` | 1 iteration rel 1e-10 (abs 1e-15); 10 iterations rel 1e-9, abs 1e-9 (the summation order separates the paths at about 1e-13) |
| `test_tsne_affinities_calibrate_every_row` | p2a property: 300 x 20 standard-normal data, perplexity 30, and the data x 10 and x 0.01: every conditional row sums to 1, its entropy is within 1e-5 of ln 30, and the rows agree across scales to 1e-4 of their largest entry | |
| `test_tsne_calibration_warns_when_unreached` | nine duplicate points cannot reach perplexity 3.9: their rows are uniform over the duplicates and the warning names 9 of 12 points and the miss ln 8 - ln 3.9 | |
| `test_tsne_affinities_have_no_floor` | cross-group entries below 1e-12, `sum(P) = 1` | |
| `test_tsne_first_step_matches_bhtsne_update` | one step from a fixed start equals the bhtsne update written out | 1e-12 |
| `test_tsne_seed_changes_the_embedding`, `test_tsne_start_is_rtsne_scale` | the seed sets the start; its standard deviation is 1e-4 | |
| `test_tsne_reports_kl_of_returned_embedding` | the reported KL is that of the returned embedding | 1e-12 relative |
| `tsne_memory_refactor_is_bit_identical` | bit-level golden of 260 iterations on a 15-point fixture (x86_64 Linux) | bits |

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::reduction::tests::test_validate_tsne ml::reduction::tests::test_tsne ml::reduction::tests::tsne
```

## MCP tool

`ml_tsne` calls `tsne` with the request's settings; its description states the defaults above
(`test_mcp_ml_descriptions_state_defaults`, `crates/p2a-mcp/tests/tier3/wp57_ml_spatial.rs`).

## References

- van der Maaten, L. and Hinton, G. (2008). "Visualizing Data using t-SNE". *JMLR*, 9, 2579-2605.
- van der Maaten, L. (2014). "Accelerating t-SNE using Tree-Based Algorithms". *JMLR*, 15, 3221-3245 (bhtsne).
- Krijthe, J. H. (2015). Rtsne: T-Distributed Stochastic Neighbor Embedding using a Barnes-Hut Implementation.
