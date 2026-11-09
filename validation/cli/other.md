# Validation: CLI causal, time-series, ML, spatial and viz commands

## Method Overview

T3-WP61 typed every option string of `p2a causal`, `p2a ts`, `p2a ml`, `p2a spatial` and
`p2a viz` (CLI-14): an option value is parsed by clap through the core option table of its
enum, so the R, CLI and MCP spellings are accepted case-insensitively ('-', '.', '_' and
spaces are equivalent) and anything else is a usage error (exit 2) that lists the accepted
values. Before, an unknown value ran a default with exit 0 (or, for `causal rd`, was refused
only after the session was opened, exit 1). The commands also print and serialise the
notes the core results carry (`Converged: ...` and `Warning: ...` lines in text; `warnings`
always, `converged` and `iterations` where reported, in JSON) and the fields the wave-2
estimators added.

The tests run the `p2a` binary (`crates/p2a-cli/tests/tier3/wp61_other.rs`) against the R
cases of the estimators' own validation (registry `validation/methods/cli_other.json`). JSON
values are compared at the `validation/README.md` tolerances for the case's n, printed text at
its printed precision, iterative fits at the tolerance of the estimator's own validation.

## Commands and pins

### `causal rd`, `causal fuzzy-rd`

| Option | Values (canonical first; aliases) | R argument |
|---|---|---|
| `--kernel` | triangular (tri), epanechnikov (epa), uniform (uni) | `rdrobust(kernel = )` |
| `--bwselect` | mserd, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1, cercomb2 | `rdbwselect(bwselect = )` |
| `--vce` | nn, hc0, hc1, hc2, hc3 | `rdrobust(vce = )` |

`fuzzy-rd` gained `--bwselect` (it always ran mserd). Both print `Kernel:` and `Bandwidth
source:` lines and serialise `bwselect`, `bandwidth_source` and `diagnostics` (identification
warnings of severity Caution and above); `warnings` leaves out the missing-value note, which
has its own `missing` key.

| Test | Case, keys | Tolerance |
|---|---|---|
| `test_validate_cli_rd_bwselect_certwo` | `rd_bwselect_sharp` `tri_p1_nn_certwo` (h and b, left and right) | `Tol::coef(1000)` |
| `test_validate_cli_rd_kernel_alias` | `rd_bwselect_sharp` `epa_p1_nn_mserd` (`--kernel epa`) | `Tol::coef(1000)` |
| `test_validate_cli_fuzzy_rd_bwselect` | `rd_bwselect_fuzzy` `fuzzy_tri_p1_nn_certwo` (`--bwselect cer_two`) | `Tol::coef(1000)` |

`test_cli_unknown_kernel_is_rejected` checks the usage errors (no R value).
