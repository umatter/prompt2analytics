# Spark handoff — code-review PR verification

**Date opened:** 2026-04-19
**Main machine (where PRs were drafted):** CPU-only, ARM64 laptop without CUDA toolkit
**This file lives on `main`.** `git fetch && git checkout main` on Spark will pull it in.

---

## Background

Five PRs implement 22 of the 24 items from `code_review_report.md`. All five branches were pushed to `origin` and are ready to review. Each is a single commit off `main`; any order of merge works.

| PR | Branch | Topic | Local verification |
|---|---|---|---|
| [#3](https://github.com/umatter/prompt2analytics/pull/3) | `security-hardening-session-1` | Path jail, SQL read-only, CORS default | ✅ build + tests green |
| [#4](https://github.com/umatter/prompt2analytics/pull/4) | `panic-elimination-session-2` | 43× `total_cmp`, StudentsT guards, SAS length, design-matrix unwrap | ✅ 1848 tests pass |
| [#5](https://github.com/umatter/prompt2analytics/pull/5) | `gpu-soundness-session-3` | Mutex-guard cuBLAS/cuSOLVER handles, SAFETY comments | ❌ **blocked — needs CUDA toolkit** |
| [#6](https://github.com/umatter/prompt2analytics/pull/6) | `api-consistency-session-4` | `LinearEstimator` for panel types, `run_gls(&Dataset, …)`, strL warning, gsynth `NotImplemented` | ✅ 1844 tests pass |
| [#7](https://github.com/umatter/prompt2analytics/pull/7) | `cleanup-session-5` | Dead code removal, `get_dataset!` macro rollout (175 sites), HDFE rayon, LLM timeouts | ✅ 57 MCP + 1842 core tests pass |

Pre-existing failures (unchanged by any PR): `test_validate_panel_fe_full_grunfeld`, `ml::mboost_fast::…_speed`, `ml::xgboost_fast::…_speed` (the last two are parallel-load timing flakes).

---

## Required: verify PR #5 on Spark

This PR changes `GpuContext` to wrap `CudaBlas` / `DnHandle` in `std::sync::Mutex` and exposes them via `GpuContext::blas()` / `GpuContext::solver()` accessors. See [context.rs in the PR](https://github.com/umatter/prompt2analytics/blob/gpu-soundness-session-3/crates/p2a-core/src/linalg/gpu/context.rs) for the full change.

### Steps

```bash
cd ~/tools/prompt2analytics    # or wherever the repo lives on Spark
git fetch origin
git checkout gpu-soundness-session-3
git pull --ff-only

# 1. Does it still compile?
cargo check -p p2a-core --features cuda 2>&1 | tee /tmp/s3-check.log

# 2. Do CUDA tests pass?
cargo test -p p2a-core --features cuda --release 2>&1 | tee /tmp/s3-test.log

# 3. Did the mutex wrapping regress performance?
cargo bench -p p2a-core --bench gpu_benchmarks --features cuda 2>&1 | tee /tmp/s3-bench.log
```

### What to look for

**Step 1 (compile) — most likely failure modes, in order of probability:**

1. **`CudaBlas: !Send`** — cudarc may not implement `Send` for `CudaBlas` / `DnHandle`. Before the PR, `unsafe impl Send + Sync for GpuContext {}` masked this. With `Mutex<CudaBlas>` as a field, the compiler will complain that `Mutex<CudaBlas>` is not `Send` because `CudaBlas` isn't `Send`.

   **Fix:** add at the top of `crates/p2a-core/src/linalg/gpu/context.rs`:
   ```rust
   // SAFETY: cuBLAS/cuSOLVER handles can be moved across threads as long as
   // the CUDA context is activated on the new thread. Mutex<T> adds the
   // serialization that the handles themselves don't provide.
   unsafe impl Send for CudaBlasWrap {}   // newtype if the foreign-type rule applies
   ```
   OR (cleaner, avoids the newtype): switch the field type to `std::sync::Mutex<Box<CudaBlas>>` or `Arc<Mutex<CudaBlas>>` if `Arc` has more lenient bounds in cudarc 0.19. **Preferred:** check whether cudarc already has `unsafe impl Send for CudaBlas` in its source before adding a workaround.

2. **`gemm` / `gemv` method resolution through `MutexGuard`** — the accessors return `MutexGuard<CudaBlas>`. Method calls like `ctx.blas().gemm(...)` rely on Deref coercion. If cudarc's `Gemm` trait has associated types that don't pass through `MutexGuard`, you may need to explicitly deref:
   ```rust
   let blas = ctx.blas();
   (*blas).gemm(cfg, ...)?;
   ```

3. **`cholesky_inverse_gpu` already ignores `_ctx`** — shouldn't need changes, but verify.

**Step 2 (tests):**
- CUDA tests were previously passing. The mutex wrapping serializes operations on the handle; no numerical change is expected.
- If any test fails, it's likely a deadlock (two operations both trying to lock the same handle in the same thread). Grep for cases where code calls `ctx.blas()` twice in a row without dropping the first guard.

**Step 3 (benchmarks):**
- Compare `/tmp/s3-bench.log` against the saved baseline at `performance/reports/gpu_performance.md`.
- Under single-threaded dispatch (the common case), there should be no regression.
- Under concurrent callers, expect serialization overhead — that's the correctness trade-off the PR is paying for.
- If `xtx` / `matmul` / k-means distances regress >10% single-threaded, something's off.

### Post results back to PR #5

Drop the relevant log tail into a comment:

```bash
gh pr comment 5 --body "$(cat <<'EOF'
Spark verification:

\`\`\`
$(tail -30 /tmp/s3-check.log)
\`\`\`

Tests: $(grep 'test result' /tmp/s3-test.log | tail -1)

Benchmarks: [paste relevant rows from /tmp/s3-bench.log]
EOF
)"
```

If the compile fails, push the fix as a new commit on `gpu-soundness-session-3` — don't amend; the reviewer wants to see the Send-workaround step explicitly.

---

## Optional: other PRs on Spark

Non-CUDA tests already passed on the main machine. Re-running on Spark only adds value if:
- Spark has a different Rust toolchain / glibc that might expose ABI issues.
- You want to revalidate MC simulations under the faster CPU (see `paper/code/mc_validation/`).
- Pre-existing `test_validate_panel_fe_full_grunfeld` failure might be ARM-specific and pass on x86_64.

If curious, run:
```bash
cargo test -p p2a-core --lib --features database 2>&1 | tail -20
cargo test -p p2a-mcp --features full 2>&1 | tail -20
```

---

## If Spark has it: run the full comparison pipeline

After all PRs merge (or cherry-picked locally), a full `performance/comparisons/run_all.sh` run would regenerate the benchmark CSVs referenced in the paper. Not required for merge — but if you're rerunning Spark-calibrated thresholds anyway, it's a cheap add:

```bash
cd performance/comparisons && ./run_all.sh --quick
```

---

## Notes on merge order

Low-risk, already-verified PRs can merge from either machine:
1. PR #3 (security) — zero conflicts expected with any other PR.
2. PR #4 (panics) — touches stats / estimator.rs / sas.rs / design.rs. No overlap with #3.
3. PR #6 (API consistency) — panel types + gls. No overlap with #3, #4.
4. PR #7 (cleanup) — touches MCP handlers that PR #3 also touches (`load_dataset`, database handlers). **Merge #3 before #7** or expect trivial conflicts in those two files.
5. PR #5 (GPU) — independent, merges whenever Spark says OK.

---

## Rollback plan if something goes wrong on merge

Each PR is a single commit. If any introduces a regression after merge:

```bash
git revert <merge-commit-sha>
```

Then reopen the session branch, fix on that branch, push, and re-merge. Do not force-push main.
