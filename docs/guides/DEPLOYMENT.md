# Deploying the web app (bring-your-own-key)

This guide covers hosting the p2a web UI publicly — e.g. a demo for reviewers —
in **bring-your-own-key (BYOK)** mode, where each visitor supplies their own LLM
API key in the browser.

## Two pieces, hosted separately

Unlike a purely static site (e.g. the Prompt Arena classroom app, which is a
static frontend + a serverless Cloudflare Worker), p2a has a **heavyweight,
stateful backend** that cannot run on Workers/Pages:

| Piece | What it is | Where it runs |
|-------|-----------|---------------|
| **Frontend** | Dioxus → static WASM + assets | GitHub Pages (free, fits the qamelab pattern) |
| **Backend** | `p2a-mcp` axum server (270 tools, SurrealDB/RocksDB, ~575 MB image) | An always-on host with HTTPS (VPS / Fly.io / Render / a box) |

The frontend half mirrors how the other qamelab sites deploy. The backend is the
genuinely new piece — it needs a real server, so it gets its own HTTPS endpoint
(e.g. `https://p2a-api.qamelab.org`) that the frontend is built to talk to.

## Frontend: GitHub Pages on a qamelab subdomain

The repo lives under `umatter/`, not the `qamelab` org, so a path like
`qamelab.org/p2a` (how Prompt Arena is served) isn't available — a **subdomain**
CNAME'd to GitHub Pages is the clean route. The `.github/workflows/web-deploy.yml`
workflow builds the WASM and publishes it; it reads two repo **Variables**
(Settings → Secrets and variables → Actions → Variables):

- `P2A_BACKEND_URL` — public HTTPS URL of the backend, baked into the WASM at
  build time (e.g. `https://p2a-api.qamelab.org`).
- `WEB_DOMAIN` — the custom domain, written to `CNAME` (e.g. `p2a.qamelab.org`).

Deploy steps:

1. Set the two repo Variables above.
2. Add a DNS record at the qamelab.org zone:
   `CNAME p2a.qamelab.org → umatter.github.io.`
3. Deploy the backend (next section) and note its HTTPS URL → that's
   `P2A_BACKEND_URL`.
4. Run the **Deploy Web App** workflow (push to `main` or trigger manually).
5. After that, the workflow also runs on every push to `main` that changes
   `crates/p2a-dioxus/**`, `Cargo.toml`, `Cargo.lock` or the workflow file, so
   merging such a change redeploys the frontend without a manual step.
6. Settings → Pages → Source: **GitHub Actions**; set the custom domain to
   `p2a.qamelab.org` and tick **Enforce HTTPS** once the cert provisions.
7. Add `https://p2a.qamelab.org` to the backend's `P2A_CORS_ORIGINS`.
8. Add the live URL to the qamelab site card (`qamelab.github.io`,
   `src/content/software/p2a.md` → `liveUrl: https://p2a.qamelab.org`).

## Deployment modes

The backend decides two things once, at startup, in
`crates/p2a-mcp/src/config.rs`: the **deployment mode**, which fixes the routes
and the browser posture, and the **resource profile** (`Off` or `Hosted`), which
fixes the limits described in [In-process controls](#in-process-controls). The
mode comes from explicit configuration, never from the bind address alone.

| | Local | Authenticated | OpenPublic |
|---|---|---|---|
| Chosen when | no token, a loopback bind and no open-mode flag; the stdio transport is always Local | `P2A_ACCESS_TOKEN` (`--access-token`) is set | no token, and `P2A_ALLOW_UNAUTHENTICATED` (`--allow-unauthenticated`) is set or the bind is not loopback (which needs that flag) |
| Resource profile | Off unless `P2A_LIMITS=hosted` | Off unless `P2A_LIMITS=hosted` | always Hosted |
| `GET /api/sessions`, `GET /api/files` | only with `--expose-enumeration` (`P2A_EXPOSE_ENUMERATION`) | registered | never |
| Host check | 421 `misdirected` unless the Host is `localhost` or a loopback IP (any port); a request without a Host header passes | off | off |
| `/ws` (feature `websocket`) | on | on | off unless `P2A_ENABLE_WS` (`--enable-ws`) |
| Per-session data directories (`P2A_SESSION_DIRS=auto`) | off | off | on |
| Server-held LLM keys | only for the provider's default endpoint, and not with the hosted profile unless `llm_server_keys=true` | same | not unless `llm_server_keys=true` (false in the hosted profile), and then only for the provider's default endpoint |

**Threat model.** There are no user accounts. A session id is a random UUID
that `POST /api/sessions` returns, and it works as a capability: whoever holds
it can use that session's datasets, files and conversations, like an unlisted
share link. In open public mode nothing lists session ids, and every object a
session owns answers 404 to any other session. A token (Authenticated mode)
restricts *who* reaches the server at all; token holders can list each other's
session ids through `GET /api/sessions`, so they trust each other.

**Startup refusals.** The process refuses to start (it exits with an error) on:

- a non-loopback bind with neither a token nor the open-mode flag;
- `P2A_LIMITS=off` (`--limits off`) in open public mode;
- `P2A_DB_PATH` inside `P2A_DATA_ROOT` in Authenticated or OpenPublic mode
  (Local mode only logs a warning, because the desktop app keeps its store under
  the user's data directory, which can lie inside a data root that defaults to
  the working directory);
- an unknown `P2A_LIMIT_OVERRIDES` key or an invalid value (the message lists
  the known keys);
- zero worker threads or concurrency, or a seconds or minutes knob above one
  year.

Two notes on choosing the mode:

- A backend bound to loopback behind a reverse proxy on the same host must run
  with a token or with the open-mode flag. In Local mode it answers 421 to every
  request whose Host is the public hostname.
- The open-mode flag on a loopback bind already selects OpenPublic and with it
  the hosted floor; a same-host proxy needs nothing else.

The stdio transport is always Local. Its request limits and per-allocation
ceiling apply when a profile is configured (`P2A_LIMITS=hosted`); deadlines,
concurrency slots and quotas exist only on the HTTP side.

The desktop app's embedded server and the paper's evaluation runs bind loopback
without a token, so they run in Local mode with no limits.

## In-process controls

These controls run inside the backend, and they are what bounds the cost of a
single request. An edge rate limit (checklist step 5) only bounds how many
requests arrive. The numbers below are the hosted defaults. With the Off
profile every HTTP tool call still runs through the governor's slots and panic
isolation, but there is no deadline, memory floor, quota or request limit unless
a knob sets one (see [Configuration reference](#configuration-reference)). The
core invariants and the file, SQL, egress and ownership checks hold in every
profile.

### Execution governor

Every tool call from `POST /api/tools/{name}`, from the LLM tool loop and from
websocket `tool` messages runs through one governor
(`crates/p2a-mcp/src/exec.rs`).

- **Off the async runtime.** Tool work runs on the blocking thread pool, and the
  runtime has by default max(2, CPUs) worker threads (`P2A_WORKER_THREADS`), so
  `/health`, admission and SSE keep-alives stay responsive on a one-vCPU machine
  while a tool computes.
- **Slots.** A call needs a slot of its session (`P2A_SESSION_CONCURRENCY`,
  hosted 1) and a global slot (`P2A_TOOL_CONCURRENCY`, default one per CPU). It
  waits at most `P2A_TOOL_QUEUE_WAIT_SECS` (hosted 10 s) for both, and is then
  refused with 429 `session_busy` (`Retry-After: 2`) or 503 `server_busy`
  (`Retry-After: 5`).
- **Deadline.** A call that runs longer than `P2A_TOOL_TIMEOUT_SECS` (hosted
  60 s) plus a 250 ms grace answers 504 `deadline_exceeded`, with
  `details.limit_secs`. The deadline also cancels the call's budget, and core
  loops stop at their next checkpoint (`p2a_core::resource::checkpoint`). A
  client that disconnects cancels its call the same way, and closing a chat
  stream stops its tool loop.
- **Panics.** A tool that panics answers 500 `internal`; the session and the
  process survive.
- **Memory admission.** While the process RSS is above
  `P2A_MEMORY_SOFT_LIMIT_MB` (hosted 700 MiB), new calls are refused with 503
  `memory_pressure` (`Retry-After: 15`). This needs a platform that reports RSS
  (Linux); elsewhere the server warns at startup and admits every call.
- **Dataset quotas.** After a call, the in-memory size of the session's
  datasets is checked against `P2A_SESSION_DATA_MB` (hosted 200 MiB) and the
  total over all sessions against `P2A_TOTAL_DATA_MB` (hosted 500 MiB). A call
  that grew the session's data past either is rolled back (the datasets are
  restored) and answered 422 `quota_exceeded`. A call that does not grow the
  session's data is never refused.
- **Per-allocation ceiling.** p2a-core sizes caller-driven buffers through
  `resource::try_*`. A single allocation above `P2A_MAX_ALLOC_MB` (hosted
  256 MiB), or one whose size overflows, fails as a tool error instead of
  aborting the process.
- **Health and watchdog.** `/health` reports a `governor` object and answers 503
  with `"status": "degraded"` while every slot is held by work past its
  deadline. If that state lasts `P2A_STUCK_EXIT_SECS` (hosted 300 s), the
  process logs an error and exits with exit code 70, so that the platform's
  restart policy starts a fresh process. A restart discards every in-memory
  session.
- **Session capacity.** When `P2A_MAX_SESSIONS` (100) sessions exist, creating
  another evicts the least recently used session that has been idle for at
  least `P2A_SESSION_EVICT_IDLE_MINUTES` (hosted 10); a session with a running
  call is never evicted. If no session qualifies, `POST /api/sessions` answers
  503 `capacity` (`Retry-After: 60`).

What the governor does not cover: work without checkpoints keeps running, and
keeps its slot, after its deadline until it finishes (its result is
discarded); memory is checked when a call starts and at guarded allocations,
not continuously; the stdio transport has no deadline, slots or quotas.

### Request limits

`crates/p2a-mcp/src/tools/limits/` classifies every integer, float and array
field of every tool, and the numbers inside array fields. When limits are set
(the hosted profile, or overrides), each call's JSON arguments are checked
before they are deserialised, on the HTTP, LLM, websocket and stdio paths.

- **Families.** 25 families of cost parameters (replications, trees,
  iterations, folds, grid points, lags, components, list lengths, ...) each
  have one cap. Every hosted family cap is at least 1.5 times the corresponding
  tool default, so a call with default arguments is never refused.
- **Tolerances.** Convergence tolerances must be finite and at least
  `min_tolerance`.
- **Row classes.** Methods whose memory grows with the square of the row count
  (`dense_rows`), or whose run time grows with its cube (`cubic_rows`) or square
  (`quadratic_rows`), refuse larger datasets with a message that suggests
  sampling with `munge_sample`.
- **Named budgets.** Sizes the handlers compute while they run: image pixels,
  table cells, join output, generated data, export and file sizes, spatial
  links, the work of an ARIMA fit, SQL results, the conversation store and the
  LLM context.
- **Response size.** A tool's text and image output above `response_bytes`
  (16 MiB) is replaced by a refusal.

An argument, row or response-size refusal answers 422 `limit_exceeded` with
`details {parameter, requested, limit}` and a message that names the parameter,
the limit and what to change. From a sandboxed server in open mode:

```json
{"success":false,"error":"Parameter `n_boot` = 1000000 exceeds this server's limit of 10000 (replications) for `regression_bootstrap_cov`. Use a smaller value.","code":"limit_exceeded","details":{"parameter":"n_boot","requested":"1000000","limit":"10000"}}
```

A named budget that a handler checks while it runs (file size, export size,
join output, table cells, ...) answers as a tool error instead: HTTP 200 with
`data.success: false` and the same kind of message. In the LLM loop either text
reaches the model as the tool's error, so it can retry with a smaller value. On
stdio a refusal is a JSON-RPC invalid-params error (-32602) whose `data.code`
is `"limit_exceeded"`. The server log line for every refusal names the
`P2A_LIMIT_OVERRIDES` key that raises the limit (`override_key`).

### Core invariants (every mode, CLI included)

p2a-core enforces these itself, on every entry point (library, CLI, stdio and
HTTP), and they never depend on the resource profile:

- parameters are checked against the data they apply to (lags and bandwidths
  below the number of observations, folds at most the number of rows,
  tolerances finite and positive, ...), and refused with an error that names the
  bound;
- buffers sized by a caller are allocated fallibly, with checked size
  arithmetic, so an impossible size is an error and not a process abort;
- loops whose progress depends on comparing values reject or filter NaN and
  infinite values first;
- method-level caps, modelled on R's own limits: multinomial and ordered models
  accept at most 100 outcome categories; C5.0 accepts `trials` from 1 to 100 and
  at most max(2, min(1000, n/2)) classes; Cubist accepts `committees` from 1 to
  100; RD local polynomials have order at most 10, and an error is returned when
  fewer than p + 2 observations lie inside the bandwidth on either side; sphet
  `instrument_order` is at most 4; density `n` is at most 2^20; panel GLS
  refuses a panel with more time periods than entities (T > N);
  `regression_bootstrap_cov` needs `n_boot` of at least 10; matching `ratio` is
  at most the number of controls;
- the C5.0, quantile random forest and Cubist tree builders stop at depth 512,
  so no input can recurse deep enough to overflow a worker's stack.

### Sessions and persistence

- **TTL.** A session expires `P2A_SESSION_TTL` minutes (60) after its last use;
  an expired session answers 410, and a cleanup task removes expired sessions
  every 10 minutes.
- **One registry.** The in-memory session table is the only record of live
  sessions. With the `db` feature, the conversation store follows it: a
  session's rows are deleted when the session is deleted, expires or is
  evicted, and a periodic sweep removes the rows of sessions that are no
  longer live and older than the TTL.
- **Ownership.** Conversations, messages, tool calls and settings answer only
  to the session that owns them. The caller names its session in the
  `X-P2A-Session-Id` header; another session's object answers 404, exactly as a
  missing one does.
- **Bounds.** A message, title or settings string longer than `message_bytes`
  (hosted 262,144 bytes) answers 413 `payload_too_large`; more than
  `conversations_per_session` (50) conversations or `messages_per_conversation`
  (1,000) messages answers 422 `quota_exceeded`.
- **History budget.** `session_db_bytes` (hosted 16 MiB, none when off) caps the
  bytes one session's conversations store: titles, message contents and
  tool-call arguments, results and errors. A client write past it answers 422
  `quota_exceeded`; text the server produces (assistant replies, tool results)
  is truncated to the room left; deleting a conversation frees its bytes.
- **Memory.** While RSS is above `P2A_MEMORY_SOFT_LIMIT_MB`, writes to the store
  answer 503 `memory_pressure` (`Retry-After: 15`).
- **`user_id`.** Optional, at most 128 bytes of `A-Z a-z 0-9 . _ @ -` (400
  `invalid_arguments` otherwise), and ignored in open public mode.

### Files

- **Path jail.** Every path a request names is resolved inside the data root
  (`P2A_DATA_ROOT`, default the working directory). A relative path resolves
  inside it; an absolute path must canonicalise inside it after `..` and
  intermediate links are resolved. Anything else is refused with a message that
  the path "is outside the data directory".
- **Links.** A path whose final component is a symbolic link is refused, live or
  dangling.
- **Writes.** Files are written to a temporary file (mode 0600) beside the
  target and renamed into place, so a link planted at the target is replaced,
  never followed.
- **Per-session directories.** In open public mode each session's root is
  `<data root>/sessions/<session id>/` (mode 0700). Other sessions' directories
  and the shared root are outside it and unreachable. The directory is removed
  with the session, and directories left by an earlier process are purged at
  the first file access after a restart. A session's directory holds at most
  `session_disk_bytes` (hosted 64 MiB).
- **Sizes.** `load_dataset` refuses a file larger than `load_file_bytes` (hosted
  64 MiB) before parsing it; `export_dataset`, `export_session` and
  `import_session` are bounded by `export_bytes` (hosted 64 MiB).
- **Messages** show paths relative to the data root, never absolute server
  paths.

### SQL

- **One statement.** Each call runs exactly one statement. A lexer that skips
  strings, quoted identifiers, dollar quotes and comments with DuckDB's rules
  finds the statement boundary and the first keyword, which must be `SELECT`,
  `WITH`, `EXPLAIN`, `PRAGMA`, `DESCRIBE` or `SHOW`. DuckDB statements also may
  not use `EXPLAIN ANALYZE` (or `EXPLAIN` with an option list), which executes
  the statement, or a `COPY` keyword anywhere outside strings, quoted
  identifiers and comments.
- **Engine lockdown.** DuckDB database files open read-only, and every DuckDB
  connection has external file access, extension installation and loading,
  community extensions and configuration changes disabled (parquet support is
  built in). SQLite databases open read-only, never as URIs, with `ATTACH`
  disabled, `query_only` on and only schema PRAGMAs admitted.
- **`db_query_file`** reads only its jailed file: the in-memory connection may
  access exactly that path, then external access is switched off and the
  configuration locked. The engine leaves that one file writable, so the
  lexical checks above are what keep it read-only.
- **Budgets.** `db_result_rows` (100,000), `db_result_bytes` (64 MiB),
  `db_timeout_secs` (20), `db_memory_mb` (256) and `db_threads` (1). A query also
  stops when the process memory grows by more than twice `db_memory_mb` while it
  runs. A result cut at the row or byte budget is stored truncated, and the tool
  output says so.
- **Free-form DuckDB SQL is off in the hosted profile.** With `sql_freeform`
  false (the hosted default), `db_duckdb_query` and `db_query_file` answer with a
  tool error reading "free-form SQL is not available on the public demo; load the
  file with load_dataset instead", before any path is checked or any engine
  opened. `db_sqlite_query` and the table and schema tools are unaffected. An
  operator who sets `sql_freeform=true` re-enables the two tools and takes on the
  DuckDB residual risk listed under
  [What this does NOT give you](#what-this-does-not-give-you).

With limits off the lockdown still applies; only the size and time budgets do
not.

### LLM egress and keys

- **Allowlist.** The LLM routes connect only to hosts in
  `P2A_LLM_ALLOWED_HOSTS` (bare hostnames, matched exactly and
  case-insensitively). When the variable is unset and `llm_allow_loopback` is
  false (the hosted default), the list defaults to api.openai.com,
  api.anthropic.com, openrouter.ai. When the variable is unset and
  `llm_allow_loopback` is true (the Off default, or a hosted server that sets
  `llm_allow_loopback=true` to reach a same-host Ollama), there is no default
  list and any public https host is allowed.
- **Scheme.** `https` only; `http` is accepted for a loopback Ollama endpoint
  and only when `llm_allow_loopback` is true (Off profile).
- **Resolve, check, pin.** Host names are resolved by the client's own
  resolver; if any address is private, loopback, link-local, shared (CGNAT),
  multicast, reserved or a documentation range (including an IPv4 address
  embedded in an IPv6 transition form), the whole host is refused; the only
  exception is the loopback Ollama endpoint described above. The client
  connects only to the addresses it checked. IP-literal hosts are checked the
  same way.
- **No redirects, no proxy.** Redirects are never followed and system proxies
  are ignored. DNS lookup and connect each time out after 10 s;
  `llm_timeout_secs` (hosted 120 s) bounds a non-streaming call in total and the
  idle time between reads of a streaming one. A provider response body is read
  up to 8 MiB.
- **Retries.** Only connect errors raised before the request was sent, HTTP 429
  and HTTP 5xx are retried, at most 3 attempts in total, honouring
  `Retry-After` up to 30 s. A timed-out completion is never sent again.
- **Upstream errors** reach the caller as `<provider> returned HTTP <status>;
  <hint>`, never with the provider's response body.
- **Server keys.** `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` from the server's
  environment is used only when the profile allows server keys
  (`llm_server_keys`, false in the hosted profile) **and** the request's
  `base_url` is unset or the provider's own default endpoint. Otherwise the
  caller must send `api_key`.
- **One chat per session.** In the hosted profile a session runs one chat
  request at a time (`llm_streams_per_session`); a second answers 429
  `session_busy` (`Retry-After: 2`).
- **Tool loop.** A request's `max_tool_iterations` (default 25) is clamped to
  `llm_max_tool_iterations` (25 hosted, 50 off). Each tool result in the loop is
  cut to at most 16 KiB. The model may not call the server's internal tools (session
  export and import, seeds, cleaning sessions, the search meta-tools, ...) or
  unknown names; registered analytical tools outside the advertised subset stay
  callable.
- **Bounded context.** The dataset block in the system prompt holds at most
  `llm_context_bytes` (24,576 bytes hosted), 20 datasets and 60 columns per
  dataset; names and sample values are JSON-quoted, and the prompt tells the
  model that the block is data, not instructions. The history sent to the model
  is cut to `llm_history_bytes` (131,072 bytes hosted).

### HTTP surface

- **Body cap.** Request bodies above `P2A_MAX_HTTP_BODY_MB` (32 MiB) answer 413
  with a plain-text body.
- **CORS** is the outermost layer, so refusals carry CORS headers. With
  `P2A_CORS_ORIGINS` it allows exactly those origins and the request headers
  `authorization`, `content-type`, `accept` and `x-p2a-session-id`, and exposes
  `Retry-After`. Without origins and without permissive mode there is no
  cross-origin access.
- **`/ws`** exists only with the `websocket` feature, is off in open public mode
  unless `P2A_ENABLE_WS` is set, runs every tool message through the governor
  and the request limits, and caps messages and frames at the body cap.
- **Token.** With `P2A_ACCESS_TOKEN`, every route except `/health` and CORS
  preflight needs `Authorization: Bearer <token>`; without it the answer is 401,
  plain text, with `WWW-Authenticate: Bearer`.

## Configuration reference

Every knob is a command-line flag with an environment variable, except the
ones marked "env only". "Off" and "Hosted" are the defaults under each resource
profile. For the timeout, watchdog, memory, data, allocation and eviction knobs
(`--tool-timeout-secs`, `--stuck-exit-secs`, `--memory-soft-limit-mb`,
`--session-data-mb`, `--total-data-mb`, `--max-alloc-mb`,
`--session-evict-idle-minutes`) an explicit `0` means none or off.

| Flag | Env | Off | Hosted | Notes |
|---|---|---|---|---|
| `--transport` | `P2A_TRANSPORT` | `stdio` | `stdio` | `http` for the server |
| `--host` | `P2A_HOST` | `127.0.0.1` | same | a non-loopback bind needs a token or open mode |
| `--port` | `P2A_PORT` | `8080` | same | |
| `--cors-origins` | `P2A_CORS_ORIGINS` | none | same | comma-separated origins |
| `--cors-permissive` | `P2A_CORS_PERMISSIVE` | false | same | development only; the env accepts only `true`/`false` |
| `--access-token` | `P2A_ACCESS_TOKEN` | none | same | selects Authenticated mode |
| `--allow-unauthenticated` | `P2A_ALLOW_UNAUTHENTICATED` | off | same | selects OpenPublic mode; the env accepts `1`/`true`/`yes`/`on` |
| `--expose-enumeration` | `P2A_EXPOSE_ENUMERATION` | off | same | Local mode only; the env accepts `1`/`true`/`yes`/`on` |
| `--enable-ws` | `P2A_ENABLE_WS` | off | same | OpenPublic only; the env accepts `1`/`true`/`yes`/`on` |
| `--session-dirs` | `P2A_SESSION_DIRS` | `auto` | `auto` | `auto` = on only in OpenPublic; `on`, `off` |
| `--session-ttl-minutes` | `P2A_SESSION_TTL` | 60 | 60 | minutes since last use |
| `--max-sessions` | `P2A_MAX_SESSIONS` | 100 | 100 | |
| `--db-path` | `P2A_DB_PATH` | none (in-memory store) | same | feature `db`; must lie outside `P2A_DATA_ROOT` outside Local mode |
| `--audit-log` | `P2A_AUDIT_LOG` | false | false | the env accepts only `true`/`false` |
| `--audit-path` | `P2A_AUDIT_PATH` | `p2a-audit.log` in the working directory | same | |
| `--limits` | `P2A_LIMITS` | none (the mode decides) | same | `hosted` or `off`; `off` is refused in OpenPublic |
| `--limit-override` | `P2A_LIMIT_OVERRIDES` | none | none | `key=value` list, see the next table |
| `--worker-threads` | `P2A_WORKER_THREADS` | max(2, CPUs) | max(2, CPUs) | async runtime workers |
| `--tool-concurrency` | `P2A_TOOL_CONCURRENCY` | CPUs | CPUs | tool calls at once, server-wide |
| `--session-concurrency` | `P2A_SESSION_CONCURRENCY` | CPUs | 1 | tool calls at once per session |
| `--tool-queue-wait-secs` | `P2A_TOOL_QUEUE_WAIT_SECS` | 300 | 10 | wait for a slot before 429/503 |
| `--tool-timeout-secs` | `P2A_TOOL_TIMEOUT_SECS` | none | 60 | per-call deadline |
| `--stuck-exit-secs` | `P2A_STUCK_EXIT_SECS` | off | 300 | watchdog, exit code 70 |
| `--memory-soft-limit-mb` | `P2A_MEMORY_SOFT_LIMIT_MB` | none | 700 | RSS admission |
| `--session-data-mb` | `P2A_SESSION_DATA_MB` | none | 200 | dataset MiB per session |
| `--total-data-mb` | `P2A_TOTAL_DATA_MB` | none | 500 | dataset MiB over all sessions |
| `--max-alloc-mb` | `P2A_MAX_ALLOC_MB` | none | 256 | largest single allocation |
| `--session-evict-idle-minutes` | `P2A_SESSION_EVICT_IDLE_MINUTES` | never | 10 | eviction when the table is full |
| `--auth-enabled` | `P2A_AUTH_ENABLED` | | | deprecated, no effect (feature `auth`) |
| `--jwt-secret` | `P2A_JWT_SECRET` | | | deprecated, no effect (feature `auth`) |
| env only | `P2A_MAX_HTTP_BODY_MB` | 32 | 32 | request bodies and websocket messages, MiB; must be a positive whole number; `0` or an invalid value means the default 32 (there is no unlimited setting) |
| env only | `P2A_DATA_ROOT` | working directory | same | must exist; resolved once, on first use |
| env only | `P2A_LLM_ALLOWED_HOSTS` | none (any public host) | api.openai.com, api.anthropic.com, openrouter.ai | comma-separated bare hostnames |
| env only | `P2A_AUDIT_MAX_MB` | 50 | 50 | audit log rotation size; `0` = never rotate; keeps 3 old files |
| env only | `RUST_LOG` | `p2a_mcp=info` | same | log filter; logs go to stderr |
| env only | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | used for default endpoints | not used (`llm_server_keys=false`) | see the server-key rule above; leave unset on a public server |

### `P2A_LIMIT_OVERRIDES` keys

Values are comma-separated `key=value` pairs (the flag can also be repeated);
a later pair wins. `none` or `unlimited` removes a limit, except for the five
keys that always have a value (`message_bytes`, `llm_timeout_secs`,
`llm_max_tool_iterations`, `llm_context_bytes`, `llm_history_bytes`); boolean
keys take `true` or `false`;
`min_tolerance` takes a positive number; every other key takes a whole number.
Byte keys are in bytes (64 MiB = 67108864), `*_mb` keys in MiB. An unknown key
or an invalid value fails startup, and the message lists the known keys. Every
hosted family cap is at least 1.5 times the corresponding tool default, so
tool defaults are never refused.

Families (Off: none):

| Key | What it bounds (typical fields) | Hosted |
|---|---|---|
| `replications` | bootstrap and permutation counts (`n_boot`, `n_bootstrap`, `n_permutations`, `bootstrap`) | 10,000 |
| `simulation_size` | Monte Carlo sizes (`n_simulations`, `n_samples`) | 100,000 |
| `trees` | `n_trees`, `ntree`, `committees` | 5,000 |
| `boosting_iterations` | `mstop`, `trials`, `nterms`, boosting `n_iter` | 10,000 |
| `mcmc_draws` | `n_draws`, `burn_in`, `ndpost`, `nskip` | 20,000 |
| `folds` | `cv_folds`, `n_folds`, `nfolds` | 20 |
| `iterations` | `max_iter`, `max_iterations`, `maxit` | 100,000 |
| `restarts` | `n_init`, `n_starts` | 100 |
| `grid_points` | `n_grid`, density `n`, `n_points`, `nlambda` | 10,000 |
| `bins` | `bins`, `breaks`, `n_bins` | 1,000 |
| `lags` | `lag_max`, `lags`, `max_lag`, `order_max`, `differences` | 1,000 |
| `horizon` | forecast `horizon`, `steps`, `n_ahead`, `h` | 5,000 |
| `model_order` | ARIMA, GARCH and VAR orders, RD `p`, `degree`, `instrument_order` | 20 |
| `bandwidth` | integer HAC, Driscoll-Kraay, GMM and sphet bandwidths | 1,000 |
| `span_width` | `spans` elements, `runmed` `k`, windows | 1,001 |
| `components` | `n_components`, `n_factors`, `rank`, `n_clusters`, k-means `k` | 500 |
| `neighbors` | kNN `k`, `n_neighbors` | 100 |
| `generated_rows` | generated and sampled rows (`n_rows`, `munge_sample` `n`, `arima_sim` `n`) | 1,000,000 |
| `image_dimension` | chart `width`, `height` | 4,000 |
| `list_length` | column lists (`x`, `covariates`, `columns`, ...); duplicates are refused | 500 |
| `array_length` | numeric arrays (`xout`, `values`, `probs`, `taus`, ...) | 100,000 |
| `string_bytes` | generator strings, labels, pad and fill strings, regex patterns (UTF-8 bytes) | 4,096 |
| `tree_depth` | `max_depth`, `maxdepth` | 100 |
| `power_sample_size` | the sample size `n` of `power_t_test` and `power_anova_test`, per group in two-sample and ANOVA designs (the top of the range in which they solve for `n`); `power_prop_test` is not capped. Do not raise it (see below) | 1,000,000 |
| `design_groups` | `power_anova_test` `groups`. Do not raise it (see below) | 10 |

Until the core bounds the t and F quantiles of the power calculations, the
`power_sample_size` and `design_groups` caps also keep those quantiles within
the range the core computes. Raising either with `P2A_LIMIT_OVERRIDES` can let
`power_t_test` or `power_anova_test` run past the deadline.

Row classes and tolerance (Off: none):

| Key | What it bounds | Hosted |
|---|---|---|
| `dense_rows` | dataset rows for `ml_hierarchical`, `ml_cutree`, `ml_dbscan`, `regression_smooth_spline` | 5,000 |
| `cubic_rows` | dataset rows for `ml_tsne`, `ml_cmdscale`, `sar_model`, `sem_model`, `sar_probit_model`, `sem_probit_model`, `spatial_panel_ml`, `spatial_panel_gmm` | 2,000 |
| `quadratic_rows` | dataset rows for `propensity_matching`, `heterogeneity_test`, `ml_svm`, `regression_loess`, `ts_changepoint` | 50,000 |
| `min_tolerance` | smallest convergence tolerance (`tol`, `eps`, ...) | 1e-15 |

Named budgets:

| Key | What it bounds | Hosted | Off |
|---|---|---|---|
| `image_pixels` | chart width × height | 16,000,000 | none |
| `table_cells` | cells of computed tables and square matrices (contingency tables, `stats_loglin`, pairwise results, `linalg_toeplitz`) | 1,000,000 | none |
| `max_groups` | groups compared pairwise | 100 | none |
| `join_output_rows` | predicted output rows of a join | 2,000,000 | none |
| `onehot_levels` | levels a one-hot encoding expands | 500 | none |
| `str_split_parts` | parts `str_split` produces | 100 | none |
| `str_pad_width` | target width of string padding (no current tool reads it) | 10,000 | none |
| `generated_cells` | cells `generate_random_data` produces | 10,000,000 | none |
| `generated_bytes` | estimated bytes of generated data | 200 MiB | none |
| `export_bytes` | `export_dataset`, `export_session` and `import_session` sizes | 64 MiB | none |
| `load_file_bytes` | size of the file `load_dataset` reads, checked before parsing | 64 MiB | none |
| `response_bytes` | one tool's text and image output | 16 MiB | none |
| `session_disk_bytes` | bytes in one session's data directory | 64 MiB | none |
| `spatial_links` | total links of a `spatial_neighbors` structure | 2,000,000 | none |
| `arima_fit_work` | work of an ARIMA fit, observations × ((p + q + 3)² + d), checked before the fit starts; while set, a series holding a nonzero value below 1e-100 is refused as well | 400,000 | none |
| `db_result_rows` | rows a SQL tool keeps | 100,000 | none |
| `db_result_bytes` | estimated bytes a SQL tool keeps | 64 MiB | none |
| `db_timeout_secs` | run time of one SQL statement | 20 | none |
| `db_memory_mb` | DuckDB memory limit (the RSS guard allows twice this) | 256 | none |
| `db_threads` | DuckDB threads | 1 | engine default |
| `sql_freeform` | whether `db_duckdb_query` and `db_query_file` run at all | `false` | `true` |
| `message_bytes` | one conversation message, title or settings string | 262,144 | 2,097,152 |
| `conversations_per_session` | conversations one session holds | 50 | none |
| `messages_per_conversation` | messages one conversation holds | 1,000 | none |
| `session_db_bytes` | bytes one session's conversation history holds | 16 MiB | none |
| `llm_timeout_secs` | provider call time (total, or idle for streams) | 120 | 900 |
| `llm_max_tool_iterations` | clamp of a request's `max_tool_iterations` | 25 | 50 |
| `llm_context_bytes` | dataset block in the system prompt | 24,576 | 65,536 |
| `llm_history_bytes` | conversation history sent to the model | 131,072 | 524,288 |
| `llm_streams_per_session` | chat requests one session runs at once | 1 | none |
| `llm_server_keys` | whether server-held provider keys may be used | `false` | `true` |
| `llm_allow_loopback` | whether a loopback Ollama endpoint may be used; `true` also drops the default host allowlist when `P2A_LLM_ALLOWED_HOSTS` is unset | `false` | `true` |

The hosted `arima_fit_work` of 400,000 allows ARIMA(1,1,1) on 15,384
observations, and on a release build the slowest fit it allows is bounded at
about 16 s; the 1e-100 floor keeps that bound, because arithmetic on numbers
that small takes a much slower path. The margin to the deadline assumes the
default 60 s: with a lower `P2A_TOOL_TIMEOUT_SECS`, or on a throttled shared
CPU, lower `arima_fit_work` in proportion with `P2A_LIMIT_OVERRIDES`. Raising it
above about 3.5 times the hosted value lets a fit run past the deadline and keep
its slot, since the fit has no checkpoint.

## Must-do checklist

1. **Do NOT set LLM API keys on the backend.**
   Leave `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` unset in
   the backend's environment. The hosted profile does not use server-held keys
   (`llm_server_keys=false`); the reason not to set them is that the same
   environment without the hosted profile (an authenticated or local deployment
   without `P2A_LIMITS=hosted`), or with
   `P2A_LIMIT_OVERRIDES=llm_server_keys=true`, would use them for every request
   to a provider's default endpoint that omits a key, which makes the server an
   open proxy billed to your account. Unset = true BYOK.

2. **Pin the LLM egress allowlist (`P2A_LLM_ALLOWED_HOSTS`).**
   Set `P2A_LLM_ALLOWED_HOSTS=api.openai.com,api.anthropic.com,openrouter.ai`
   (bare hostnames, matched exactly and case-insensitively against the URL
   host). The hosted profile uses this list when the variable is unset, but only
   while `llm_allow_loopback` stays false; set it explicitly anyway, so the
   policy depends on neither the profile nor that flag. An Authenticated or
   Local server on the Off profile without it can reach any public https host
   a caller names, carrying whatever key the caller supplied. With or without
   the list, every resolved address is still checked and internal addresses are
   refused, except a loopback Ollama endpoint while `llm_allow_loopback` is
   true (the Off default).

3. **Terminate TLS in front of the backend.**
   BYOK keys travel in request bodies; without HTTPS they're exposed on the wire.
   Run a reverse proxy (Caddy/nginx/Traefik) or a platform that provides HTTPS.

4. **Lock CORS to your frontend origin.**
   Set `P2A_CORS_ORIGINS=https://your-demo.example` (comma-separated for
   several), and list every browser origin that sends the access token or the
   session header. Never use `--cors-permissive` / `P2A_CORS_PERMISSIVE` in
   production.

5. **Add an edge rate limit as one layer, not as the control.**
   A per-IP rate limit at a reverse proxy or CDN adds what the process cannot
   do: it slows session-creation churn, chat starts and API scraping from one
   address, and it brings TLS and a WAF at the edge. It does not bound the cost
   of a single request (the governor and the request limits do), it does not
   see the messages inside one websocket connection (a connection counts as one
   request), and it does not stop a client that spreads work over many
   sessions; only the global tool slots bound that. A CDN in front (e.g.
   **Cloudflare**, free tier) works well and is invisible to reviewers; a
   reverse proxy (Caddy/nginx) rule works too. **Exclude the SSE path**
   `/api/llm/chat/stream` from response buffering and idle timeouts, or
   streaming chat will break.

   Putting Cloudflare in front does not break the SSE path, as long as you do
   not disable the server's keep-alive: the stream emits an axum keep-alive
   every 15s against a 900s Proxy Idle Timeout, and the 125s Proxy Read Timeout
   applies to the response *headers*, which the handler sends immediately. A
   long tool loop therefore does not trip a 524. If you ever remove the
   keep-alive, that stops being true.

   `[http_service.concurrency]` in `deploy/fly.toml` is a proxy-side ceiling on
   in-flight requests. It bounds open connections, not work: SSE streams hold a
   slot for their whole life, and one admitted request can still be expensive.
   Tool work is bounded inside the process by the execution governor
   (`P2A_TOOL_CONCURRENCY`, one call per session in the hosted profile, 503
   `server_busy` or 429 `session_busy` with `Retry-After` after
   `P2A_TOOL_QUEUE_WAIT_SECS`).

   **The rule, for a Cloudflare Free plan.** Free gives you exactly **one**
   rate limiting rule, counted per *IP* (the characteristic is fixed), with a
   period of 10s or 1min, and *Block* as the only action. So it has to be one
   broad rule rather than a per-endpoint set:

   - Expression: `starts_with(http.request.uri.path, "/api/") or http.request.uri.path eq "/ws"`
     (the `/ws` part matters only if you enable websockets; `/ws` is off in
     open mode by default)
   - When rate exceeds: **60 requests / 1 minute** (per IP)
   - Action: *Block*

   That threshold is deliberately generous for a human and still cuts off a
   script, because **a browser session makes far fewer requests than it looks
   like**. The LLM tool loop runs server-side: one `POST /api/llm/chat/stream`
   can drive many tool calls without any of them crossing the edge. A reviewer
   uploading data and chatting generates single-digit requests per minute.

   Note that an SSE stream counts as **one** request no matter how long it stays
   open, so it does not need excluding from a *rate* limit — that is a different
   concern from excluding it from response buffering and idle timeouts, which it
   does need (see above). Tighten the threshold if you ever see abuse; the cost
   of being wrong is a reviewer seeing Cloudflare error 1015.

6. **Cap request body size.**
   `P2A_MAX_HTTP_BODY_MB` (default 32) limits memory from oversized uploads and
   also caps websocket messages and frames.

7. **Confine the filesystem jail (`P2A_DATA_ROOT`).**
   Set `P2A_DATA_ROOT` to a dedicated, **empty** directory (e.g. `/data`, which
   the backend image creates) so file and database tools can only reach that
   directory. Unset, it defaults to the process working directory — never leave
   it defaulting to a home directory on an exposed host. The directory must
   exist. In open mode `sessions/` under it is reserved for the per-session
   directories, and files the operator places in the shared root are not
   reachable by visitors. The disk worst case is `P2A_MAX_SESSIONS` ×
   `session_disk_bytes`, 100 × 64 MiB = 6.25 GiB with the hosted defaults; keep
   that product within the disk the host has, and lower `session_disk_bytes`
   with `P2A_LIMIT_OVERRIDES` if needed.

8. **Keep the session database outside the jail (`P2A_DB_PATH`).**
   File tools can write anywhere under `P2A_DATA_ROOT`, so the server refuses
   to start in Authenticated or OpenPublic mode with `P2A_DB_PATH` inside it.
   `docker-compose.yml` uses `/var/lib/p2a/db` on its own `p2a-db` volume. A
   store left in the old `p2a-data` volume by an earlier compose file must be
   moved there or deleted.

9. **Run the container locked down.**
   `docker-compose.yml` ships with `cap_drop: ALL`, `no-new-privileges`,
   `pids_limit`, and memory/CPU limits. Writable state is confined to the
   `p2a-data` volume (`/data`), the `p2a-db` volume (`/var/lib/p2a`) and a
   `/tmp` tmpfs; enable `read_only: true` after verifying those are the only
   write paths.

10. **Keep the hosted floor.**
    Run `P2A_LIMITS=hosted` on any server that untrusted or semi-trusted
    callers reach, including one whose access token is shared with reviewers.
    Raise single limits with `P2A_LIMIT_OVERRIDES` or the governor flags, never
    with `P2A_LIMITS=off`.

## API-key handling (for transparency)

User keys are **never persisted or logged** server-side. Each request constructs
a per-request provider, sets one outbound auth header — `Authorization: Bearer`
for OpenAI-compatible providers including OpenRouter, `x-api-key` for Anthropic
— then drops the key. There is no server-side key store (the former unused
`settings.api_key_encrypted` field was removed). The keys' only exposure point
is **in transit** — hence the TLS requirement above.

**Server-held keys.** The backend uses its own `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` only when the profile allows server keys and the request
targets that provider's default endpoint; a server key is never sent to a host
the caller picked. With the hosted profile's `llm_server_keys=false` it uses
none of them, and `GET /api/llm/env-keys` answers `false` for both providers.

**Upstream errors.** A provider's error body is never returned to the caller;
the caller sees the provider, the HTTP status and a hint. At most 512
characters of the body are logged, at `debug` level.

**Transit is web-only.** Desktop and mobile builds start an embedded backend on
loopback, so the key never leaves the device there. On a hosted deployment it
does pass through your backend, on every route that carries a provider config:
`/api/llm/chat`, `/api/llm/chat/stream`, and `/api/llm/generate-title`. The
frontend says so — "sent with each request through our backend to your chosen
provider" — so keep this section and that copy in agreement.

**Client-side storage.** The web app keeps the API key and the backend access
token in the browser's `localStorage` for its origin, in plaintext. The desktop
app writes keys typed into it to `p2a-settings.json` in its configuration
directory with mode 0600; keys it reads from `*_API_KEY` environment variables
are used but never written. Neither uses an OS keychain. JSON exports from the
web app carry no session or backend ids.

**Access token.** For a server started with `P2A_ACCESS_TOKEN`, the web app
sends the token from its "Backend access token" setting, and `p2a-chat` takes
`--access-token` (or `P2A_ACCESS_TOKEN`). The browser origin that sends it must
be listed in `P2A_CORS_ORIGINS`.

**The destination is only as constrained as `P2A_LLM_ALLOWED_HOSTS`.** Callers
supply their own `base_url`, so on an exposed deployment set this variable (step
2 above); the default list applies only when it is unset and
`llm_allow_loopback` is false (the hosted default).

**Keys are the smaller half of the story.** Every tool call also sends the
user's dataset, prompt, and tool arguments to your server — that is the larger
disclosure, and it is inherent to running the analytics server-side. Tool
arguments are not written to disk unless you set `P2A_AUDIT_LOG`; leave it
unset on a demo, or say plainly that you have enabled it.
[DATA_SECURITY.md](DATA_SECURITY.md) describes what the audit log records.

## Example: Caddy reverse proxy

```caddy
your-demo.example {
    encode gzip
    # TLS (automatic) and unbuffered SSE only. Add rate limiting with a Caddy
    # plugin or at a CDN in front. Websocket upgrades are proxied as they are.
    @stream path /api/llm/chat/stream
    reverse_proxy @stream backend:8080 {
        flush_interval -1   # disable buffering for SSE
    }
    reverse_proxy backend:8080
}
```

## What this does NOT give you

The open-but-hardened mode is appropriate for an **open demo with disposable,
isolated sessions**. It is *not* a multi-tenant SaaS: there are no user
accounts and no server-side key storage. The in-process controls bound what
one request can cost; these risks remain:

- **Session ids are capabilities.** They are unguessable and never listed in
  open mode, but treat a leaked session URL like a leaked share link, and treat
  server logs as sensitive: they record session ids at `info` level.
- **Any restart loses everything.** A deploy, a watchdog exit or an
  out-of-memory kill discards every in-memory session, dataset and
  conversation.
- **The session table can be kept full.** A client that creates sessions
  faster than the eviction threshold frees them can keep new visitors out
  (503 `capacity`). The edge rate limit bounds this. `P2A_MAX_SESSIONS` is cheap
  to raise for memory, because the byte quotas bound data, but it multiplies the
  disk worst case.
- **Cancellation is cooperative.** Work without checkpoints keeps its slot
  after its deadline until it finishes; the watchdog exists for that case.
- **Memory is checked at call start and at guarded allocations.** One call can
  still allocate past the machine's memory in a path the guards do not cover,
  for example while a file is parsed before the dataset quota is checked.
- **Free-form DuckDB SQL.** Where it runs (the Off profile, or
  `sql_freeform=true`), DuckDB can build very large values within one result
  chunk, beyond its own memory limit, before the server's checks see them. The
  hosted default turns those tools off for this reason.
- **No global disk quota.** Only each session's directory is bounded.
- **No per-IP fairness inside the process.** One address with many sessions
  competes for the same global slots as everyone else.
- **Steered models.** An LLM steered by text in the data can call any tool
  within its own session, but cannot reach another session or files outside the
  session's directory.
- **stdio has no deadline.** Calls over the stdio transport run to completion.

If you need to restrict *who* can use it, switch to the authenticated posture
(`P2A_ACCESS_TOKEN`), keep `P2A_LIMITS=hosted`, and distribute the token; if
you need real multi-tenant accounts and server-side key storage, add an identity
layer in front of the backend (KMS-managed master key — see the discussion in
project memory).

## Release and redeploy

These are the maintainer's steps for shipping a new backend to the hosted demo.
Nothing here runs automatically except the two workflows named in steps 0 and
2.

0. **Order.** Merging to `main` redeploys the web app automatically (when the
   merge touches the paths listed in the frontend section). The current web app
   sends `X-P2A-Session-Id` on the conversation routes and an older backend
   ignores the header, so the web app can go first. The reverse is not true: a
   new backend answers 400 to a conversation-detail request that names no
   session, which an older web app sends. Release the backend soon after the
   merge, and do not deploy a new backend behind an old web app.
1. **Version.** Choose `X.Y.Z`; for 0.x releases a minor bump (for example
   `0.2.0`) marks changed HTTP behaviour, but the choice is yours. Edit
   `[workspace.package] version` in `Cargo.toml`, then refresh the lockfile:

   ```bash
   cargo update --workspace
   git diff --stat Cargo.lock      # only the five p2a-* entries may change
   git commit -am "chore: bump workspace version to X.Y.Z"
   ```
2. **Tag.** The tag push is what builds and publishes the backend image (and the
   desktop bundles, `.github/workflows/desktop-bundle.yml`):

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main vX.Y.Z
   ```

   `.github/workflows/backend-image.yml` then publishes `X.Y.Z`, `vX.Y.Z` and
   `sha-<short>` to GHCR. Do not run that workflow by `workflow_dispatch` on
   `main` for a release: a dispatch on `main` publishes only `latest` and a sha
   tag.
3. **Confirm the image exists.** The package is public, so an anonymous token
   is enough:

   ```bash
   TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:umatter/prompt2analytics/backend:pull" | jq -r .token)
   curl -s -H "Authorization: Bearer $TOKEN" https://ghcr.io/v2/umatter/prompt2analytics/backend/tags/list | jq .tags
   ```

   The list must contain `X.Y.Z`.
4. **Re-pin.** Set `[build].image` in `deploy/fly.toml` to
   `ghcr.io/umatter/prompt2analytics/backend:X.Y.Z` and commit
   `chore(deploy): pin backend image to X.Y.Z`.
5. **Validate and deploy.**

   ```bash
   fly config validate --strict -c deploy/fly.toml
   fly deploy -c deploy/fly.toml
   ```

   A deploy restarts the machine and discards every in-memory session, so pick
   a quiet time.
6. **Verify**, with read-only requests:
   - `curl -s https://p2a-api.qamelab.org/health | jq '{status, version, governor}'`
     shows `X.Y.Z` and a `governor` object;
   - `GET /api/sessions` answers 405, `GET /api/files` 404 and `GET /ws` 404;
   - one chat in the web app works, and its conversation list reloads.
7. **Roll back** by re-pinning the previous tag and running `fly deploy` again.
8. **Dependency checks.** The Security audit workflow runs weekly and on every
   change to `Cargo.lock`, any `Cargo.toml` or `audit.yml`. "Dependabot security
   updates" is a repository setting that only the owner can turn on.

## Fly.io quick reference

`deploy/fly.toml` is configured for the open-but-hardened posture: it pins an
explicit image tag and sets open mode (`P2A_ALLOW_UNAUTHENTICATED=1`),
`P2A_LIMITS=hosted`, `P2A_DATA_ROOT=/data`, the CORS origin, the LLM host
allowlist and a restart-on-failure policy. To ship a new backend build, follow
[Release and redeploy](#release-and-redeploy).

Put Cloudflare (or your CDN) in front of `p2a-api.qamelab.org` with a Rate
Limiting rule as one layer (checklist step 5).

**Keep the backend hostname one level deep.** It is `p2a-api.qamelab.org` and not
`api.p2a.qamelab.org` for a reason: on a Cloudflare full setup, Universal SSL
covers the apex and *first-level* subdomains only, and a `*.example.com`
wildcard does not match `api.staging.example.com`. Proxying a second-level host
therefore leaves the edge with no certificate to present, and it aborts the TLS
handshake outright — the origin stays healthy while every client gets a
handshake failure, and `curl -k` fails too, since nothing is served to
distrust. Covering a deeper name needs Advanced Certificate Manager, Total TLS,
or an uploaded wildcard. Renaming is free; those are not.

Fly issues its own certificate for the origin, so add the hostname
DNS-only first (`fly certs add`, then A/AAAA records with the proxy off) and
only enable the proxy once `fly certs check` reports `Issued`. Verify TLS on the
new hostname *before* repointing `P2A_BACKEND_URL` at it.
