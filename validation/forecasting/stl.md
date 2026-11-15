# STL decomposition

`stl`, `run_stl` and `run_stl_with_config` in `crates/p2a-core/src/forecasting/stl.rs` (`StlConfig`).
R reference case: `ts_utils_t4` (`validation/r/cases/ts_utils_t4.R`), series `y` of
`validation/datasets/generated/ts_utils_t4.csv` (n = 144, monthly, level about 1,000 to 1,290, sd 90.57).

## Method

STL (Cleveland et al. 1990) through the `stlrs` crate, a port of the original Fortran.

| Rule | p2a | R `stats::stl` |
|---|---|---|
| Arithmetic | `stlrs` computes in f32. The series is centred by its mean and scaled by its standard deviation (a constant series is only centred) before the call; the components are restored afterwards (T = s T' + c, S = s S') | double precision throughout |
| Remainder | x - S - T in f64 | `as.vector(x) - seasonal - trend` |
| Seasonal window | `seasonal_length` (odd, >= 3); default nextodd(period) (`stlrs`) | `s.window`, no default |
| Periodic window | `StlConfig::periodic()`: seasonal window 10 n + 1, seasonal degree 0, the seasonal component replaced by its cycle means | `s.window = "periodic"`, the same three steps |
| Other windows, degrees, loops | `stlrs`'s defaults, R's formulas (trend window nextodd(ceiling(1.5 period / (1 - 1.5 / s.window))), low-pass nextodd(period), inner 2 (1 robust), outer 0 (15 robust)) | the same |

LOESS commutes with an affine map of the series, and the robustness weights depend only on ratios of remainders, so
centring and scaling change no component beyond the f32 rounding. `periodic()` with `seasonal_length` or
`seasonal_degree` is `InvalidSpecification`.

## Pins

| Test | R call | Pinned |
|---|---|---|
| `test_validate_stl_periodic_ts_utils_t4` | `stl(ts(y, frequency = 12), s.window = "periodic")` | seasonal, trend, remainder (144 values each) |
| `test_validate_stl_swin13_ts_utils_t4` | `stl(ts(y, frequency = 12), s.window = 13)` (13 = p2a's default) | the same |
| `test_validate_stl_level_1e8_ts_utils_t4` | both calls on `y + 1e8` | the same |

**Tolerance.** Absolute 1e-5 * sd(y) = 9.06e-4: stlrs computes in f32, so after centring and scaling the components
carry about 1e-7 of the series' spread per operation. R's components at level 1e8 agree with those at level 0 to about
1e-8. Before the centring, p2a's f32 path quantised a remainder at level 1e8 to multiples of 8.

`test_stl_reconstructs_exactly` (T + S + R = x to 1e-9 of the level at 1e8), `test_stl_periodic_with_seasonal_length_is_refused`
and `test_stl_constant_series` hold the properties; `test_stl_recovers_dgp` is a property on a known process, not an R pin.

**Not yet in the CLI.** `p2a ts stl` has no seasonal-window option; the MCP server has no STL tool.

## References

- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). "STL: A Seasonal-Trend Decomposition
  Procedure Based on Loess". *Journal of Official Statistics*, 6(1), 3-73.
- R Documentation: `stats::stl()`.
