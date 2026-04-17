# Code Review Report — prompt2analytics Workspace

**Reviewer:** Claude Sonnet 4.6 (automated)  
**Date:** 2026-04-17  
**Scope:** `crates/p2a-core/src/`, `crates/p2a-mcp/src/`, `crates/p2a-cli/src/`, `crates/p2a-chat/src/`, `crates/p2a-dioxus/src/`  
**Excluded:** `validation/`, `paper/`, `performance/`, `target/`, `docs/`

---

## 1. Executive Summary

The workspace is technically functional and architecturally coherent: the p2a-core/p2a-mcp split is clean, the tool-handler pattern is consistent, and the GPU-acceleration fallback design is sound. However, five material risks exist:

1. **Path traversal / SQL injection with no server-side access control.** The `/api/files` browser and `load_dataset` / database tools accept arbitrary paths from HTTP clients without canonicalization or jail enforcement. With CORS set to permissive by default in the embedded-server configuration, any web page can probe the local filesystem.

2. **Widespread panicking sorts on user-supplied floating-point data.** Approximately 37 call sites use `sort_by(|a, b| a.partial_cmp(b).unwrap())` on data that may contain NaN or Inf coming from user datasets. This terminates the entire server process.

3. **`unreachable!()` reachable at runtime in `treatment.rs`.** The AIPW bootstrap branch is statically unreachable as written today, but it is one refactor away from becoming a live panic if the match arm structure is ever extended.

4. **`GmmResult`, `PanelGlsResult`, and `PvcmResult` do not implement `LinearEstimator`** despite being the primary output of major estimators. Callers cannot use the trait-based pipeline (confidence intervals, AIC/BIC, export helpers) with these types.

5. **The MCP server ships with no authentication by default.** The `auth` feature is optional and off by default (`default = []`). The embedded-library configuration forces `cors_permissive: true`. A publicly reachable deployment runs with zero access control.

---

## 2. Correctness Concerns

### 2.1 Panicking sort on NaN-containing user data

**Files:**  
- `crates/p2a-core/src/stats/spline.rs:137, 238, 299, 344`  
- `crates/p2a-core/src/stats/mood.rs:181`  
- `crates/p2a-core/src/stats/fligner.rs:307, 320`  
- `crates/p2a-core/src/stats/robust.rs:51, 125, 190, 197, 260, 516`  
- `crates/p2a-core/src/stats/ansari.rs:189`  
- `crates/p2a-core/src/stats/spectrum.rs:1227`  
- Plus ~20 additional sites in ML and econometrics modules

Each of these calls `sort_by(|a, b| a.partial_cmp(b).unwrap())` on `f64` vectors. `partial_cmp` returns `None` for NaN, causing the `unwrap()` to panic. Some call sites (e.g., `robust.rs:44`, `mood.rs:147`) filter NaN before the sort, but many do not—in particular all four sites in `spline.rs` and both in `fligner.rs`. Since these functions accept user-supplied dataset columns, a column containing a single NaN value will crash the MCP server process.

Note: `spline.rs:129` checks `is_finite` for single-domain interpolation, but lines 235 and 341 (approximation functions) filter for `is_finite` and then sort the *filtered* slice at lines 238 and 344 respectively—those unwraps are safe. Lines 137 and 299 (the natural and Hermite spline interpolants) do *not* pre-filter.

### 2.2 `unreachable!()` in live match arm of `treatment.rs`

**File:** `crates/p2a-core/src/econometrics/treatment.rs:1193`

The bootstrap loop for `run_doubly_robust` is guarded by `DRMethod::IPW | DRMethod::Regression =>`, so `DRMethod::AIPW` never reaches line 1193 today. However, the code path at lines 1150–1202 uses a nested `match config.method { ... DRMethod::AIPW => unreachable!() }` inside the `IPW | Regression` arm's closure. This compiles without warning because the outer arm excludes AIPW, but it is a maintenance trap: if a future developer splits the outer arm or adds a new variant, the `unreachable!()` panics in production. The cleaner fix is to eliminate the inner match entirely (it is inside an arm that already excludes AIPW).

### 2.3 Panicking distribution constructors in `estimator.rs`

**File:** `crates/p2a-core/src/traits/estimator.rs:114, 137, 226, 241, 256, 265, 272, 280`

`StudentsT::new(0.0, 1.0, df).unwrap()` is called at multiple sites. The `statrs` constructor returns `Err` if `df <= 0.0` or is NaN. The callers do guard `df <= 0.0` at lines 109 and 132, but the guards produce a short-circuit return *before* the `unwrap`, so the unwrap is safe on those paths. However, `t_test_p_value()` (line 226) guards `df > 0.0` then immediately constructs the distribution with `unwrap()`. If `df` is `f64::NAN` or `f64::INFINITY` it passes the `> 0.0` guard but the `statrs` constructor may still reject it. The same pattern appears in `chi_squared_p_value` (line 241) and `f_test_p_value` (line 256). These are public utility functions called throughout the codebase.

Additionally, sites in `stats/mcnemar.rs:134`, `stats/kruskal.rs:220`, `stats/fligner.rs:210`, `stats/proptest.rs:429`, `stats/friedman.rs:235`, `stats/ttest.rs:544, 576`, `stats/pairwise.rs:680`, and `ml/ctree.rs:321, 366` all use the same unwrapping pattern. Some of these are well-guarded; others receive degrees-of-freedom values computed from user data without NaN checks.

### 2.4 `series.get(i).unwrap()` on unchecked index access in `design.rs`

**File:** `crates/p2a-core/src/linalg/design.rs:197`

```rust
let str_values: Vec<String> = (0..series.len())
    .map(|i| format!("{:?}", series.get(i).unwrap()))
    .collect();
```

`series.get(i)` returns `Result<AnyValue, PolarsError>`. The `unwrap()` can panic if `i` is out of bounds (impossible here since `i < series.len()`) but also if there is a cast error. While in practice this loop is safe, it silently discards cast errors as panics. Using `map_err` and propagating the error would be more robust.

### 2.5 `try_into().unwrap()` in SAS binary parser

**File:** `crates/p2a-core/src/data/sas.rs:471, 472`

```rust
ByteOrder::LittleEndian => f64::from_le_bytes(buf.try_into().unwrap()),
ByteOrder::BigEndian    => f64::from_be_bytes(buf.try_into().unwrap()),
```

`buf` is a `vec![0u8; 8]` with bytes read starting at `8 - length`. If `length` is 0 or greater than 8 due to a malformed file, the slice has the wrong length and `try_into()` returns `Err`, causing a panic. This is in the binary file parser that accepts arbitrary `.sas7bdat` files from users.

### 2.6 Incomplete bootstrap in `synth.rs::bootstrap_gsynth`

**File:** `crates/p2a-core/src/econometrics/synth.rs:3548`

```rust
// TODO: Implement full bootstrap
Ok((None, None, None))
```

`bootstrap_se: true` in `GsynthConfig` silently returns no standard errors. The caller receives `(None, None, None)` with no warning to the user. This is a correctness issue: users enabling bootstrap inference get the same output as users who disabled it.

### 2.7 Stata strL columns silently replaced with placeholder

**File:** `crates/p2a-core/src/data/stata.rs:382-387`

Stata `.dta` files using `strL` (long string) type return the literal string `"<strL>"` for every value instead of the actual string content. No warning is emitted. Users loading files with long-string columns will see corrupted data without indication.

### 2.8 HDFE column demeaning is sequential, not parallel

**File:** `crates/p2a-core/src/econometrics/hdfe.rs:339–348`

`demean_matrix_map` processes `k` columns in a sequential loop. Each column runs the full MAP convergence loop independently. For large panel datasets with many regressors, parallelizing this loop with `rayon::par_iter` over columns would be straightforward and would cut runtime by `k`-fold for the dominant demeaning cost.

---

## 3. API Consistency

### 3.1 `LinearEstimator` trait implemented only by `OlsResult`

**File:** `crates/p2a-core/src/regression/ols.rs:197`  
**Missing implementations:**  
- `crates/p2a-core/src/econometrics/panel/dynamic_panel.rs` — `GmmResult`  
- `crates/p2a-core/src/econometrics/panel/gls_models.rs` — `PanelGlsResult`  
- `crates/p2a-core/src/econometrics/panel/heterogeneous.rs` — `PvcmResult`  
- `crates/p2a-core/src/econometrics/panel/types.rs` — `PanelResult`

All four of these structs carry `coefficients: Vec<f64>`, `std_errors: Vec<f64>`, `t_stats: Vec<f64>`, `p_values: Vec<f64>`, `n_obs: usize`, etc., which are exactly what `LinearEstimator` requires. Because they do not implement the trait, callers cannot use the uniform confidence-interval, AIC/BIC, or LaTeX export pipeline defined on the trait. The `PanelResult` Display impl (`types.rs:71`) duplicates logic already in `OlsResult`'s Display.

### 3.2 Inconsistent `run_*` signatures across the regression family

- `run_ols(dataset: &Dataset, y_col: &str, x_cols: &[&str], ...)` — Dataset-based API  
- `run_quantreg(dataset: &Dataset, y_col: &str, x_cols: &[&str], ...)` — Dataset-based API  
- `run_gls(y: &[f64], x: &[f64], n_cols: usize, ...)` — raw-slice API  

`run_gls` (`crates/p2a-core/src/regression/gls.rs:373`) does not accept a `Dataset` reference, breaking the uniform `run_*` pattern. MCP handlers must manually extract arrays before calling it, adding conversion boilerplate inconsistent with every other estimator.

### 3.3 Duplicated `format_diagnostic_warnings` function

**Files:**  
- `crates/p2a-mcp/src/server.rs:548` (defined, **never called**)  
- `crates/p2a-mcp/src/tools/handlers/causal.rs:63` (defined and used)

These two functions have the same name, similar purpose, but different filtering behavior (the `server.rs` version filters to `>= Caution`; the `causal.rs` version shows all warnings) and different output formatting. The `server.rs` version is dead code.

### 3.4 229 handler sites not using `get_dataset!` macro

`crates/p2a-mcp/src/tools/common.rs` defines `get_dataset!` and `get_dataset_mut!` macros for uniform dataset lookup with consistent error messages. However, `panel.rs`, `ml.rs`, and others perform inline `match datasets.get(...)` at 229 call sites. This produces slightly different error messages across handlers and duplicates ~4 lines per lookup.

---

## 4. Security

### 4.1 No path restriction on `/api/files` endpoint (path traversal)

**File:** `crates/p2a-mcp/src/transport/http.rs:424–496`

```rust
let path = match &query.path {
    Some(p) if !p.is_empty() => PathBuf::from(p),
    _ => dirs::home_dir().unwrap_or_else(|| PathBuf::from("/")),
};
let entries = match std::fs::read_dir(&path) { ... }
```

The user-supplied path is used directly with no canonicalization and no jail enforcement. A client can request `?path=/etc` or `?path=/root` and receive a directory listing of system files. The endpoint is exposed on the HTTP transport without authentication when `auth` feature is not compiled in. The hidden-file filter (`starts_with('.')`) does not prevent access to `/etc/passwd`, `/etc/ssh/`, or database credential files.

### 4.2 No restriction on paths passed to `load_dataset`, `db_sqlite_query`, `db_duckdb_query`

**Files:**  
- `crates/p2a-mcp/src/tools/handlers/data.rs:70`  
- `crates/p2a-core/src/data/database.rs:75, 234`  
- `crates/p2a-mcp/src/tools/requests/database.rs:15, 33, 59, 75`

`load_dataset` accepts `request.path` as a `String` and passes it directly to `DataLoader::load(PathBuf::from(&request.path))`. `db_sqlite_query` and `db_duckdb_query` accept `db_path: String` with no validation. No call site canonicalizes the path or restricts it to an allowed directory. An LLM or malicious client can read any file accessible to the server process.

### 4.3 SQL query passed directly to database without restriction

**File:** `crates/p2a-core/src/data/database.rs:83, 244`

```rust
let mut stmt = conn.prepare(query)?;
```

The full SQL query string from `SqliteQueryRequest.query` is sent verbatim to `rusqlite::Connection::prepare`. While SQLite prepared statements prevent parameter injection, DML statements (`DROP TABLE`, `DELETE FROM`, `INSERT INTO`) are accepted without restriction. The tool schema comment says "SELECT statements only recommended" but nothing enforces this. For SQLite this has limited blast radius (the file is already accessible), but `query_duckdb` accepts any query against a DuckDB connection that can read arbitrary files via `COPY`, `read_csv_auto`, etc.

### 4.4 `cors_permissive: true` is the default for embedded server

**File:** `crates/p2a-mcp/src/lib.rs:47`

```rust
impl Default for EmbeddedServerConfig {
    fn default() -> Self {
        Self {
            cors_permissive: true,
            ...
        }
    }
}
```

The embedded library mode (used by the Dioxus app) defaults to `cors_permissive: true`, which sets `Access-Control-Allow-Origin: *`. Combined with the filesystem and database access issues above, any web page served from any origin can trigger file reads, database queries, and tool execution against the local server.

### 4.5 `auth` feature off by default; `full` feature does not warn

**File:** `crates/p2a-mcp/Cargo.toml`

```toml
[features]
default = []
full = ["http", "websocket", "auth", "llm", "db"]
```

`default = []` means the server compiles without authentication when users run `cargo build -p p2a-mcp`. Only `full` includes `auth`. The CLAUDE.md development guide uses `--features full` consistently, but a naive production deployment that omits features gets an unauthenticated server. There is no compile-time or runtime warning when `http` is enabled without `auth`.

### 4.6 Audit log `unwrap()` in production code path

**File:** `crates/p2a-mcp/src/audit.rs:219`

```rust
let json = serde_json::to_string(&entry).unwrap();
```

This is inside `AuditLogger::log()`. `serde_json::to_string` can fail for values containing non-UTF-8 data or maps with non-string keys. An audit entry whose `arguments` field contains such data would panic the logging task. While `AuditEntry` is a controlled struct with known types, the `arguments: serde_json::Value` field is copied directly from user tool call input and could theoretically contain values that cause serialization failure in edge cases.

---

## 5. Performance & Memory

### 5.1 `ResultCache` is exported but never used in production code

**Files:**  
- `crates/p2a-core/src/cache.rs` (implementation)  
- `crates/p2a-core/src/lib.rs:253` (re-exported)

`ResultCache`, `CacheKey`, and `CacheStats` are fully implemented (300+ lines) and publicly exported. No production code path in `p2a-core`, `p2a-mcp`, or `p2a-cli` uses them. The cache exists only in its own unit tests and a benchmark file. This is dead code that carries maintenance cost.

### 5.2 HDFE multi-column demeaning is sequential

**File:** `crates/p2a-core/src/econometrics/hdfe.rs:339–348`

`demean_matrix_map` iterates columns with `for j in 0..k`. Each column's `demean_map` call is independent. For `k` regressors with `max_iterations` up to 10,000, parallelizing with `rayon::par_bridge` over `0..k` would linearly reduce demeaning time. The function currently allocates per-column scratch buffers inside `demean_map` (line 255), which already avoids aliasing; columns can be processed in parallel without conflict.

### 5.3 LLM providers use `Client::new()` with no timeout

**Files:**  
- `crates/p2a-mcp/src/llm/openai.rs:27`  
- `crates/p2a-mcp/src/llm/anthropic.rs` (similar)  
- `crates/p2a-mcp/src/llm/ollama.rs` (similar)

All three providers create a bare `reqwest::Client::new()` with no `timeout()` or `connect_timeout()` configured. A slow or hung LLM endpoint will block the async task indefinitely, consuming an async worker. This can cause request queues to grow unbounded under sustained load or with a misconfigured API endpoint.

### 5.4 `demean_map` clones the entire input vector unnecessarily for single-factor case

**File:** `crates/p2a-core/src/econometrics/hdfe.rs:248–261`

```rust
if factors.is_empty() {
    return (data.clone(), 0, 0.0, true);
}
...
if factors.len() == 1 {
    let mut result = data.to_vec();
    demean_by_factor_inplace(&mut result, &factors[0], &mut sums_bufs[0]);
    return (Array1::from_vec(result), 1, 0.0, true);
}
```

The early return clones `data` even in the zero-factor case where the result is the input unchanged. Returning a reference or using `Cow` would avoid the allocation, which matters for large panels where `n` may be in the millions.

### 5.5 `build_history()` in Dioxus allocates a new `Vec<Message>` on every SSE event

**File:** `crates/p2a-dioxus/src/state/chat.rs:411`

`build_history()` is called on each streaming token to assemble the history for multi-turn context. It allocates a new `Vec<Message>` and clones all message content, tool call arguments, and tool results on every call. For a conversation with many messages and large tool results (truncated at 2000 chars each), this is O(messages * token_count) total allocation. Caching the history vector and rebuilding only on message completion would reduce this to O(messages).

---

## 6. Code Organization & Duplication

### 6.1 `format_diagnostic_warnings` duplicated between `server.rs` and `causal.rs`

See §3.3. The `server.rs` version (line 548) is dead code and should be removed. The `causal.rs` version should be moved to a shared utility module in `tools/handlers/mod.rs` so other handlers (`regression.rs`, `panel.rs`) can use it if they add diagnostic integration.

### 6.2 229 inline dataset lookup match blocks

See §3.4. The `panel.rs` and `ml.rs` handlers do not use the `get_dataset!` macro defined for this purpose, leading to 229 boilerplate match blocks. Each is approximately 6 lines and generates a slightly different error message string.

### 6.3 `causal.rs` handler is 2891 lines

**File:** `crates/p2a-mcp/src/tools/handlers/causal.rs`

This is the largest handler file. It covers IV, DiD, staggered DiD, Bacon, ETWFE, IPW, doubly robust, double ML, CBPS, WeightIt, entropy balancing, SBW, TWANG, matching, TMLE, CTMLE, gformula, longitudinal TMLE, and mediation — all in one file. Following the existing split pattern used for `discrete/` (which is a directory), this should be split into at least `iv.rs`, `did.rs`, `weighting.rs`, and `tmle.rs`.

### 6.4 `#[allow(dead_code)]` on `ssr` and `sst` fields in `OlsResult`

**File:** `crates/p2a-core/src/regression/ols.rs:189, 193`

`ssr` and `sst` are marked `#[allow(dead_code)]`. They are computed and stored but never read outside of construction. Either expose them (they are useful for hypothesis tests) or remove them. Suppressing the dead code warning without resolution accumulates technical debt.

### 6.5 Multiple `#[allow(dead_code)]` in `hdfe.rs` precomputed fields

**File:** `crates/p2a-core/src/econometrics/hdfe.rs:463–475`

Five fields in a struct are marked `#[allow(dead_code)]`. These appear to be precomputed values intended for future use. Either use them or remove them.

### 6.6 WebSocket LLM chat is a stub

**File:** `crates/p2a-mcp/src/transport/websocket.rs:275–282`

The `ClientMessage::Chat` handler returns a stub response: `"Chat functionality coming in Phase 3."` The WebSocket feature is included in `full` and is compiled into the production binary. Users who connect via WebSocket and attempt chat will receive a misleading non-error response. This should either be removed from the feature set or return a proper `not_implemented` error type.

---

## 7. Testing Gaps

### 7.1 Zero tests in all 18 MCP tool handler files

All files in `crates/p2a-mcp/src/tools/handlers/` have zero `#[test]` functions. The handlers contain the logic that maps LLM/client requests to `p2a-core` calls, including column name extraction, config construction, and output formatting. Bugs in this layer produce silent wrong output that is invisible to the `p2a-core` tests. Even basic smoke tests that confirm a handler returns `CallToolResult::success` for a minimal valid input would catch most handler regressions.

### 7.2 `p2a-cli` has zero tests

**File:** `crates/p2a-cli/src/`

The CLI has 0 `#[test]` functions across all source files. The session recording, script export, and output formatting logic (`session.rs`, `output.rs`) have no coverage. The CLI is a user-facing binary and regressions in argument parsing or output formatting go undetected.

### 7.3 Complex econometric modules with only 5–8 tests relative to size

| File | Lines | Tests | Ratio |
|------|-------|-------|-------|
| `synth.rs` | 3673 | 14 | 1/262 |
| `survival.rs` | 3047 | 14 | 1/217 |
| `ctmle.rs` | 2383 | 14 | 1/170 |
| `gformula.rs` | 1568 | 5 | 1/314 |
| `splm.rs` | 2241 | 5 | 1/448 |

These are numerically complex methods where correctness is hard to verify by eye. The existing tests appear to be primarily smoke tests (does the function run) rather than correctness tests (does the output match known values within tolerance). `gformula.rs` with 1568 lines and only 5 tests is particularly exposed.

### 7.4 `data/loader.rs` has only 2 tests covering 7 file formats

**File:** `crates/p2a-core/src/data/loader.rs`

The `DataLoader` handles CSV, Parquet, Excel (xlsx/xls/xlsb/ods), Stata (.dta), and SAS (.sas7bdat). Only 2 tests exist (`test_load_csv_string` and one more). The SAS and Stata parsers in particular are complex binary format parsers that are essentially untestable without test fixtures in the repo.

---

## 8. Documentation Gaps

### 8.1 `#![warn(missing_docs)]` is commented out

**File:** `crates/p2a-core/src/lib.rs:224`

```rust
// #![warn(missing_docs)]
```

This lint is disabled, so the majority of public functions lack rustdoc comments. The crate has 270 public econometric methods; an LLM or MCP client introspecting `cargo doc` would find most functions undocumented.

### 8.2 `unsafe` blocks in GPU BLAS have no `SAFETY` comments

**Files:**  
- `crates/p2a-core/src/linalg/gpu/blas.rs:82, 123, 169`  
- `crates/p2a-core/src/linalg/gpu/kernels.rs:103`  
- `crates/p2a-core/src/linalg/gpu/solver.rs:133`

Each `unsafe { ctx.blas.gemm(...) }` block calls into cuBLAS via FFI. The safety invariants (device memory must be allocated, the stream must be in the correct state, the leading dimension must match the matrix layout) are not documented. The row-major to column-major transposition convention is documented at the module level but not at the call site.

### 8.3 `unsafe impl Send` and `unsafe impl Sync` for `GpuContext` have inadequate justification

**File:** `crates/p2a-core/src/linalg/gpu/context.rs:32–35`

```rust
// Safety: CUDA handles are thread-safe when used with proper stream synchronization.
// cuBLAS and cuSOLVER handles are internally synchronized.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}
```

The comment is a claim, not a proof. cuBLAS handles are *not* thread-safe without external synchronization — the cuBLAS documentation requires that a handle only be used from one thread at a time unless the application provides its own serialization. Since `GpuContext` is accessed via a `OnceLock<Option<GpuContext>>` and shared across threads, concurrent `gemm` calls on the same handle require the cuBLAS handle itself to be protected by a mutex. The current implementation has no such mutex; concurrent GPU operations from different Rayon threads could corrupt state.

---

## 9. Dependencies

### 9.1 `MemoryTracker` is defined but never used in MCP

`crates/p2a-core/src/memory.rs` implements `MemoryTracker` with configurable memory limits (line 109). The MCP server uses `MemoryProfiler` (tracking only) but not `MemoryTracker` (enforcement). No dataset loading path enforces a memory ceiling. A client can load an arbitrarily large file and exhaust server memory.

### 9.2 `ResultCache` is dead code (see §5.1)

The cache implementation serializes results to JSON strings and stores them in a `HashMap`. This is functional but uses no actual LRU data structure — the eviction at `evict_lru()` (line 278) iterates the entire `HashMap` to find the minimum `last_accessed` timestamp, which is O(n) per eviction. If the cache were actually used at scale, this would be a performance issue. Given it is unused, the entire module should be removed.

### 9.3 No version pinning audit noted as a concern

Workspace dependencies use caret versions (`"0.52"`, `"0.16"`, etc.) which is appropriate for a library. No obviously outdated pinned versions were identified that represent a security concern.

---

## 10. Action Plan

---

### Action 1
- **File:** `crates/p2a-mcp/src/transport/http.rs:428–431`
- **Severity:** critical
- **Problem:** `/api/files` accepts arbitrary user-supplied paths with no canonicalization or jail, enabling full-filesystem directory traversal via the HTTP API.
- **Fix:** Canonicalize the requested path with `std::fs::canonicalize()` and compare it with `starts_with()` against a configured allowed root (e.g., the home directory or a configurable `P2A_DATA_ROOT` env var). Return `403 Forbidden` if the path escapes the allowed root. Add the `data_root` field to `HttpConfig`.
- **Verification:** `curl http://localhost:8080/api/files?path=/etc` should return 403; `curl http://localhost:8080/api/files?path=/home/user` should succeed.

---

### Action 2
- **File:** `crates/p2a-mcp/src/tools/handlers/data.rs:70`
- **Severity:** critical
- **Problem:** `load_dataset` passes user-supplied `request.path` directly to `DataLoader::load` with no path validation, allowing the MCP client to read any file on the filesystem.
- **Fix:** Apply the same canonicalization and allowed-root check as Action 1. Introduce a helper `fn validate_data_path(p: &str, root: &Path) -> Result<PathBuf, McpError>` and call it at the top of `load_dataset`, `export_dataset`, and the database tool handlers.
- **Verification:** Passing `path = "/etc/passwd"` to `load_dataset` should return an error response, not file contents.

---

### Action 3
- **File:** `crates/p2a-core/src/data/database.rs:83, 244`
- **Severity:** critical
- **Problem:** SQL queries from `db_sqlite_query` and `db_duckdb_query` are executed verbatim. DuckDB's `query_duckdb` can run `COPY`, `read_csv_auto('/etc/passwd')`, and other file-reading SQL that bypasses the filesystem restriction on `load_dataset`.
- **Fix:** Add a simple statement-type check before execution: parse the first non-whitespace token (case-insensitively) and reject anything not in `{SELECT, WITH, EXPLAIN, PRAGMA}`. Return `DatabaseError::Forbidden(query)` for DML and DDL. Alternatively, enforce read-only mode: for SQLite use `SQLITE_OPEN_READONLY`; for DuckDB open with `duckdb::AccessMode::ReadOnly`.
- **Verification:** `SELECT * FROM read_csv_auto('/etc/passwd')` via `db_duckdb_query` should return an error.

---

### Action 4
- **File:** `crates/p2a-mcp/src/lib.rs:47`
- **Severity:** critical
- **Problem:** `EmbeddedServerConfig::default()` sets `cors_permissive: true`, meaning any web page can make authenticated-equivalent requests to the locally running server.
- **Fix:** Change the default to `cors_permissive: false`. Add a doc comment explaining that `cors_permissive` should only be used in local development or when the server is bound to loopback only. Add a startup warning (via `tracing::warn!`) if `cors_permissive: true` is combined with a non-loopback bind address.
- **Verification:** Starting the embedded server with defaults and requesting from a cross-origin page should receive CORS errors.

---

### Action 5
- **File:** `crates/p2a-core/src/linalg/gpu/context.rs:34–35`
- **Severity:** critical
- **Problem:** `unsafe impl Sync for GpuContext` is incorrect: cuBLAS handles require external serialization for concurrent calls. Concurrent Rayon threads calling GPU operations on a shared handle can corrupt state.
- **Fix:** Wrap `blas: CudaBlas` in a `tokio::sync::Mutex` (or `std::sync::Mutex` if used from non-async contexts) inside `GpuContext`. Change `pub blas: CudaBlas` to `pub blas: Mutex<CudaBlas>`. Update all callers to lock before GEMM calls. Alternatively, if single-threaded GPU use is sufficient, store the `GpuContext` in a thread-local rather than a global `OnceLock`.
- **Verification:** Run `cargo test -p p2a-core --features cuda --release -- --test-threads=8` and verify no data races under ThreadSanitizer.

---

### Action 6
- **File:** `crates/p2a-core/src/stats/spline.rs:137, 299` and `crates/p2a-core/src/stats/fligner.rs:307, 320` and `crates/p2a-core/src/stats/mood.rs:181`
- **Severity:** critical
- **Problem:** `sort_by(|a, b| a.partial_cmp(b).unwrap())` panics when data contains NaN. These functions receive user-supplied dataset columns.
- **Fix:** Replace each occurrence with `sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))` or, better, validate that the input data is finite before sorting and return `EconError::InvalidInput` if NaN values are present (which gives the user actionable feedback rather than a crash).
- **Verification:** `interpolate_spline(&[f64::NAN, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.5])` should return `Err(...)`, not panic.

---

### Action 7
- **File:** `crates/p2a-core/src/stats/robust.rs:51, 125, 190, 197, 260, 516` and ~20 additional sites
- **Severity:** critical
- **Problem:** Same as Action 6: `sort_by(...unwrap())` on data that may contain NaN despite some call sites filtering first. The filtering is inconsistent across the ~37 sites.
- **Fix:** Audit all 37 sites (identified via `grep -rn "partial_cmp.*\.unwrap()\b"`). For each site, either (a) add a pre-sort NaN filter with an error return, or (b) use `unwrap_or(Equal)`. Standardize to option (a) for functions accepting user data, and (b) for internal functions where NaN is impossible by construction.
- **Verification:** Pass a column with `f64::NAN` to `fivenum`, `iqr`, `median_absolute_deviation`, and `ecdf`; each should return `Err` rather than panic.

---

### Action 8
- **File:** `crates/p2a-core/src/traits/estimator.rs:114, 137, 226, 241, 256, 265`
- **Severity:** important
- **Problem:** `StudentsT::new(0.0, 1.0, df).unwrap()` panics if `df` is `f64::NAN` or `f64::INFINITY`. The `df > 0.0` guard at line 219 does not catch these cases.
- **Fix:** Add `|| !df.is_finite()` to each guard condition: `if df <= 0.0 || !df.is_finite() { return f64::NAN; }`. Apply the same fix to `chi_squared_p_value` and `f_test_p_value`.
- **Verification:** `t_test_p_value(1.5, f64::NAN)` and `t_test_p_value(1.5, f64::INFINITY)` should return `f64::NAN`, not panic.

---

### Action 9
- **File:** `crates/p2a-core/src/data/sas.rs:468–472`
- **Severity:** important
- **Problem:** `buf.try_into().unwrap()` panics on malformed SAS files where `length` is 0 or > 8, causing the MCP server to crash.
- **Fix:** Replace the `unwrap()` with `map_err(|_| SasError::ParseError("Truncated double has invalid length".to_string()))?` and propagate the error. Add a bounds check: `if length == 0 || length > 8 { return Err(SasError::ParseError(...)); }` before the vector allocation.
- **Verification:** Pass a crafted `.sas7bdat` with a truncated double of length 9; the function should return `Err`, not panic.

---

### Action 10
- **File:** `crates/p2a-core/src/econometrics/treatment.rs:1193`
- **Severity:** important
- **Problem:** `DRMethod::AIPW => unreachable!()` inside a nested match is a maintenance trap. If the outer arm is ever refactored, this becomes a live panic.
- **Fix:** Remove the inner match entirely. The outer `DRMethod::IPW | DRMethod::Regression =>` arm already guarantees only those two methods reach the block. The nested re-match is redundant. Replace with a simple `if config.method == DRMethod::IPW { ... } else { ... }` for the two cases.
- **Verification:** `run_doubly_robust` with `DRMethod::AIPW` and `bootstrap > 0` should not change behavior after the refactor.

---

### Action 11
- **File:** `crates/p2a-core/src/econometrics/synth.rs:3544–3549`
- **Severity:** important
- **Problem:** `bootstrap_se: true` in `GsynthConfig` silently returns `(None, None, None)` — no SEs are computed, no warning is emitted, and the caller receives `Option::None` confidence intervals without knowing they were disabled.
- **Fix:** Either implement the bootstrap (resample donor pool assignments) or emit a `tracing::warn!` and return a `Result::Err(EconError::NotImplemented("gsynth bootstrap SE"))` so callers know the configuration was not honored.
- **Verification:** `GsynthConfig { bootstrap_se: true, .. }` should either produce SEs or an explicit error, not silently produce `None`.

---

### Action 12
- **File:** `crates/p2a-core/src/data/stata.rs:382–387`
- **Severity:** important
- **Problem:** Stata `strL` columns are silently replaced with the literal string `"<strL>"` with no warning. Users see corrupted string data.
- **Fix:** At minimum, emit a `tracing::warn!` when a `strL` column is encountered. Add `strL` to the documented limitations in the function's rustdoc. Tracking a `Vec<bool>` of which columns had strL values and including a warning in the `DatasetInfo` returned to the MCP layer would give the LLM/user actionable information.
- **Verification:** Load a `.dta` file containing strL columns; the `load_dataset` output should mention "strL columns not fully supported".

---

### Action 13
- **File:** `crates/p2a-core/src/econometrics/panel/` — `dynamic_panel.rs`, `gls_models.rs`, `heterogeneous.rs`, `types.rs`
- **Severity:** important
- **Problem:** `GmmResult`, `PanelGlsResult`, `PvcmResult`, and `PanelResult` carry all the data needed by `LinearEstimator` but do not implement it. The trait-based pipeline (confidence intervals, AIC/BIC, LaTeX export) is unavailable for these estimators.
- **Fix:** Implement `LinearEstimator` for each type. Note that these structs use `Vec<f64>` rather than `Array1<f64>`; either convert the fields to `Array1<f64>` (preferred for consistency with `OlsResult`) or implement `LinearEstimator` with `.into()` conversions in each accessor.
- **Verification:** `PanelResult::confidence_intervals(0.95)` should compile and return non-NaN intervals for a valid result.

---

### Action 14
- **File:** `crates/p2a-core/src/regression/gls.rs:373–386`
- **Severity:** important
- **Problem:** `run_gls` takes raw `&[f64]` slices instead of `&Dataset` + column names, inconsistent with every other `run_*` function in the regression family and requiring callers to manually extract arrays.
- **Fix:** Add `pub fn run_gls_from_dataset(dataset: &Dataset, y_col: &str, x_cols: &[&str], correlation_type: &str, correlation_param: Option<f64>) -> EconResult<GlsResult>` that extracts columns and delegates to the existing raw-slice function. Deprecate or make the raw-slice version `pub(crate)`.
- **Verification:** The MCP regression handler for GLS should not contain manual `f64` array extraction after this change.

---

### Action 15
- **File:** `crates/p2a-mcp/src/server.rs:548–579`
- **Severity:** important
- **Problem:** `format_diagnostic_warnings` in `server.rs` is defined but never called — dead code. The `causal.rs` version (which is used) has slightly different formatting. Having two versions invites divergence.
- **Fix:** Delete the `server.rs` version. Move the `causal.rs` version to `crates/p2a-mcp/src/tools/handlers/mod.rs` as `pub(super) fn format_diagnostic_warnings(...)` so it can be imported by any handler.
- **Verification:** `cargo build -p p2a-mcp` should not emit "function is never used" warning after removal.

---

### Action 16
- **File:** `crates/p2a-mcp/src/tools/handlers/panel.rs` and `crates/p2a-mcp/src/tools/handlers/ml.rs`
- **Severity:** minor
- **Problem:** 229 inline `match datasets.get(&request.dataset) { Some(ds) => ds, None => return error_text(...) }` blocks ignore the `get_dataset!` macro defined for this purpose, producing inconsistent error messages.
- **Fix:** Replace all 229 inline match blocks with `let datasets = self.datasets.read().await; let dataset = get_dataset!(datasets, &request.dataset);`. The macro exists in `tools/common.rs` and is already imported via `crate::get_dataset!`.
- **Verification:** `cargo clippy -p p2a-mcp` should report no "match can be simplified" lints; error messages from panel and ML tools should match the canonical format from the macro.

---

### Action 17
- **File:** `crates/p2a-core/src/econometrics/hdfe.rs:339–348`
- **Severity:** minor
- **Problem:** `demean_matrix_map` processes `k` independent columns sequentially. For large `k` this is the dominant cost of HDFE estimation.
- **Fix:** Replace the `for j in 0..k` loop with `(0..k).into_par_iter().map(|j| { ... }).collect::<Vec<_>>()` using rayon. Each iteration uses only its own pre-allocated scratch buffers, so there is no data dependency. Return `(x_demeaned, max_iterations, max_change, all_converged)` after aggregating with `fold`/`reduce`.
- **Verification:** `cargo test -p p2a-core -- test_validate_hdfe` should pass; runtime for `k=20` should decrease proportionally to available cores.

---

### Action 18
- **File:** `crates/p2a-core/src/linalg/gpu/blas.rs:82, 123, 169` and `kernels.rs:103` and `solver.rs:133`
- **Severity:** minor
- **Problem:** `unsafe { ctx.blas.gemm(...) }` blocks have no `// SAFETY:` comment explaining the invariants required of the caller.
- **Fix:** Add a `// SAFETY:` comment before each unsafe block documenting: (1) the device memory pointers are valid and owned by `ctx.stream`, (2) the leading dimensions match the ndarray row-major layout as described in the module doc, (3) alpha/beta are finite.
- **Verification:** `cargo clippy -- -W clippy::undocumented_unsafe_blocks` should not flag these blocks after the fix.

---

### Action 19
- **File:** `crates/p2a-mcp/src/llm/openai.rs:27` (and `anthropic.rs`, `ollama.rs`)
- **Severity:** minor
- **Problem:** `Client::new()` creates a `reqwest` client with no timeout. A hung LLM endpoint blocks indefinitely.
- **Fix:** Replace `Client::new()` with `Client::builder().timeout(Duration::from_secs(300)).connect_timeout(Duration::from_secs(30)).build().expect("Failed to build HTTP client")` in all three provider constructors. 300s accommodates long LLM completions; 30s connect timeout fails fast on network errors.
- **Verification:** Point the client at a non-responding server; the request should time out within 30 seconds.

---

### Action 20
- **File:** `crates/p2a-core/src/cache.rs`
- **Severity:** minor
- **Problem:** `ResultCache`, `CacheKey`, and `CacheStats` are fully implemented (330+ lines) and publicly exported but never used in production code. Additionally, the LRU eviction (`evict_lru`) is O(n) per eviction, making it unsuitable for actual use at scale.
- **Fix:** Either (a) remove the entire `cache.rs` module and its re-export from `lib.rs:253` and `pub mod cache` from `lib.rs:232`; or (b) if caching is planned, replace the `HashMap` + linear-scan eviction with a proper LRU structure (e.g., the `lru` crate or an `IndexMap` with LRU ordering). Document the intended use case.
- **Verification:** `cargo build -p p2a-core` should compile without "unused" warnings after removal.

---

### Action 21
- **File:** `crates/p2a-mcp/src/transport/websocket.rs:275–282`
- **Severity:** minor
- **Problem:** The WebSocket `Chat` message type returns a stub string `"Chat functionality coming in Phase 3."` instead of an error. Users receive no indication that the feature is unimplemented.
- **Fix:** Return a `ServerMessage::Error { id, message: "LLM chat over WebSocket is not yet implemented. Use the HTTP /api/llm/chat/stream endpoint.".to_string() }` instead of the stub text response.
- **Verification:** Sending a `Chat` message over WebSocket should produce a `type: "error"` response, not `type: "text"`.

---

### Action 22
- **File:** `crates/p2a-core/src/linalg/design.rs:197`
- **Severity:** minor
- **Problem:** `series.get(i).unwrap()` silently panics on Polars cast errors when extracting group identifiers.
- **Fix:** Replace with `series.get(i).map_err(|e| DesignError::Internal(format!("Cannot read group value at index {}: {}", i, e)))?` and propagate. The function already returns `Result`, so this requires changing `.map(|i| ...)` to `.map(|i| -> Result<String, DesignError> { ... }).collect::<Result<Vec<_>, _>>()?`.
- **Verification:** `extract_groups` on a DataFrame with a column that fails cast should return `Err` rather than panic.

---

### Action 23
- **File:** `crates/p2a-core/src/regression/ols.rs:189, 193`
- **Severity:** minor
- **Problem:** `ssr` and `sst` fields are computed, stored, and suppressed with `#[allow(dead_code)]`. They are never read outside construction.
- **Fix:** Expose them as public fields (they are useful for F-test and R² computation by callers) and remove the `#[allow(dead_code)]`. Alternatively, if they are truly not needed, remove the fields and their computation from `run_ols_raw`.
- **Verification:** `OlsResult::ssr` and `OlsResult::sst` should be accessible in downstream crates after the fix.

---

### Action 24
- **File:** `crates/p2a-mcp/src/audit.rs:219`
- **Severity:** minor
- **Problem:** `serde_json::to_string(&entry).unwrap()` inside the audit log path panics on serialization failure, taking down the logging task.
- **Fix:** Replace with `if let Ok(json) = serde_json::to_string(&entry) { writeln!(...); } else { tracing::error!("Failed to serialize audit entry"); }`.
- **Verification:** Audit log should not cause panics when `entry.arguments` contains unusual JSON values.

---

*End of report.*
