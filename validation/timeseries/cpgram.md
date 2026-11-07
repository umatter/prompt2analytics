# Cumulative periodogram

`cpgram`, `run_cpgram` and `white_noise_test` in `crates/p2a-core/src/forecasting/cpgram.rs`.
R reference case: `ts_filter_ar_cpt` (`validation/r/cases/ts_utils.R`), series `wn` (N(3, 1), n = 100) and `ar1`
(AR(1) 0.9 around 10, n = 120) of `validation/datasets/generated/ts_utils.csv`.

`stats::cpgram` plots and returns `NULL`, so the case writes out its body. p2a computes the same steps:

1. **Centre, then taper.** `spec.taper(scale(x, TRUE, FALSE), p = taper)`: with m = floor(n p), the first and last m values
   are multiplied by 0.5 (1 - cos(pi (2i - 1) / (2m))), i = 1..m. Tapering before centring would turn the mean into a
   low-frequency component and declare every series with a non-zero mean non-white.
2. **Periodogram.** `Mod(fft(x))^2 / n` at the frequencies k / n. R's set has floor(n / 2) + 1 points including (0, 0), and
   drops the last when that count is even. p2a reports it without (0, 0), so it holds mp points.
3. **Cumulative periodogram.** Running sum over the total. A series with no variation left is an error.
4. **Band.** q = sqrt(mp) + 0.12 + 0.11 / sqrt(mp), half-width `critical_value` = 1.358 / q, around the line 2 f, clipped
   to [0, 1].

The verdict is `is_white_noise` = max_k |C_k - 2 f_k| <= `critical_value`. `ks_statistic` is the Stephens-modified D, that
maximum times q, whose 5 % point is R's 1.358. `ks_p_value` is the asymptotic Kolmogorov survival function, without
shortcuts: 1 - (sqrt(2 pi) / x) sum_{k odd} exp(-k^2 pi^2 / (8 x^2)) below 1, and
2 sum_{k >= 1} (-1)^(k-1) exp(-2 k^2 x^2) from 1 on, where the first form loses digits.

A taper outside [0, 0.5] (R's `spec.taper`: "'p' must be between 0 and 0.5"), a non-finite value and fewer than 4
observations are errors.

## Pins

| Test | Pinned | Tolerance |
|---|---|---|
| `test_validate_cpgram_white_noise` | `wn`: crit 0.18843774417144613, max dev 0.10889787598160139, statistic 0.7847860641362071, p 0.5690823493746275, white; every frequency and cumulative value | `Tol::coef(n)` frequencies, `Tol::stat(n)` the rest, `Tol::p_value(n)` p |
| `test_validate_cpgram_ar1` | `ar1`: crit 0.1723313593732768, max dev 0.7753420099521079, statistic 6.1098250100511695, p 7.527266358277316e-33, not white | as above |
| `test_validate_periodogram_against_r` | literal R output, n = 16 | 1e-9 relative |
| `test_validate_cpgram_untapered_against_r` | literal R output, n = 65, taper 0 | 1e-10 relative |
| `test_kolmogorov_sf_matches_r` | six points of the survival function | 1e-12 relative |

`test_cpgram_rejects_bad_taper_and_non_finite` holds the refusals. Without the `spectral-analysis` feature the
periodogram is a direct DFT, which `fft_periodogram_matches_direct_dft` compares with the FFT.
