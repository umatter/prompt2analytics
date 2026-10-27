# Data Security and Privacy

This document describes where `prompt2analytics` stores data, what leaves your
machine, and how to delete it.

## Overview

Where your data goes depends on how you run p2a:

- **CLI and desktop app.** Data and results stay on your machine. The LLM
  provider you choose (a cloud API, or a local Ollama) receives your prompts and
  a summary of your datasets.
- **A self-hosted backend** (`p2a-mcp --transport http`, for example with
  Docker). Every dataset, prompt and tool argument a client sends reaches that
  server; the operator decides where it is stored (below).
- **The hosted demo** (`p2a.qamelab.org` with its backend at
  `p2a-api.qamelab.org`). Your datasets, prompts and LLM API key pass through
  our server. Do not upload confidential data.

The analytics engine itself (`p2a-core`) makes no network requests.

## Data Storage Locations

### CLI (`p2a`)

| Data Type | Location | Notes |
|-----------|----------|-------|
| Session files | the path given with `--session` | records commands for reproducibility |
| Datasets | in memory only | not persisted between runs (unless a session file is used) |
| Output files | the paths you give | charts, exports, saved datasets |
| Replay temp file | `.p2a_session_<uuid>.json` in the working directory | written by an exported script (`p2a script export`) and removed at the end of a successful run |

The CLI does not create files in hidden directories or system locations.

### MCP Server (`p2a-mcp`)

| Data Type | Location | Notes |
|-----------|----------|-------|
| Datasets | in memory, per session | gone when the session ends or the process stops |
| Files tools read and write | under `P2A_DATA_ROOT` (default: the working directory) | every path is confined to this root |
| Per-session directories (open public mode) | `<P2A_DATA_ROOT>/sessions/<session id>/`, mode 0700 | deleted with the session; left-over directories are purged at the first file access after a restart, except those of sessions a RocksDB session store can still adopt (accessed within the session TTL; at most `P2A_MAX_SESSIONS` directories, those of the most recently used sessions that have one), which go when the store deletes the session's rows or with the session once it has come back and used its files |
| Temporary files of writes | `.<file name>.<uuid>.p2a-tmp` beside the target, mode 0600 | renamed over the target when the write ends; one left by a process that stopped mid-write is removed by the first sweep of its folder after nobody has written to it for ten minutes. A tool path that resolves through the data-root jail sweeps the root (or the session's directory) and the folder the path lies in or names, the first time in a process and then at most once every ten minutes |
| Session database (feature `db`) | in-memory SurrealDB unless `P2A_DB_PATH` names a RocksDB directory | conversations, messages, tool calls, settings, dataset metadata; `P2A_DB_PATH` must lie outside `P2A_DATA_ROOT` for an authenticated or open public server |
| Audit log | off by default; `p2a-audit.log` in the working directory, or `P2A_AUDIT_PATH` | mode 0600; rotates at `P2A_AUDIT_MAX_MB` (default 50, keeps 3 old files, `0` = never rotate). Keep it outside `P2A_DATA_ROOT` |
| Server logs | stderr | filter `RUST_LOG`, default `p2a_mcp=info`; info-level lines include session ids |

### Desktop app (`p2a-dioxus`, desktop build)

| Data Type | Location | Notes |
|-----------|----------|-------|
| Session database | `p2a/data` under the user data directory: Linux `~/.local/share/p2a/data`, macOS `~/Library/Application Support/p2a/data`, Windows `%APPDATA%\p2a\data` | conversation history of the embedded server |
| Settings and API keys | `p2a-settings.json` in `p2a` under the user configuration directory: Linux `~/.config/p2a/`, macOS `~/Library/Application Support/p2a/`, Windows `%APPDATA%\p2a\` | file mode 0600, directory 0700 |
| Files tools read and write | under `P2A_DATA_ROOT`, or the app's working directory | see below |

API keys you type into the desktop app are stored in `p2a-settings.json` in
plaintext. Keys the app reads from the `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or
`OPENROUTER_API_KEY` environment variables are used but never written. The app
does not use the operating system's keychain.

The desktop app starts an embedded backend on `127.0.0.1:8081` (local mode, no
resource limits). Its file tools can reach everything under its data root, which
is `P2A_DATA_ROOT` or the app's working directory. An app started from a
launcher often has the home directory as its working directory, so set
`P2A_DATA_ROOT` to a dedicated folder.

### Web app (`p2a-dioxus`, web build)

| Data Type | Location | Notes |
|-----------|----------|-------|
| LLM settings, API key, backend access token | the browser's `localStorage` for the app's origin | plaintext; clear the site data to remove them |
| Conversation history | the backend's session database | see the server table |

### Docker / Compose

| Data Type | Location | Notes |
|-----------|----------|-------|
| Data root | `p2a-data` volume at `/data` (`P2A_DATA_ROOT=/data`) | the only directory file tools reach |
| Session database | `p2a-db` volume at `/var/lib/p2a` (`P2A_DB_PATH=/var/lib/p2a/db`) | persists across container restarts |
| Audit log | when enabled, `p2a-audit.log` in `/home/p2a` (the container's working directory) | a relative `P2A_AUDIT_PATH` resolves against `/home/p2a`; lost on container removal unless you mount a volume for it |

The image runs as the unprivileged user `p2a`. `docker-compose.yml` mounts both
volumes:

```yaml
volumes:
  - p2a-data:/data
  - p2a-db:/var/lib/p2a
```

### Hosted demo (Fly.io)

There is no volume. Datasets and conversation history live in the machine's
RAM, and session files on its ephemeral disk. All of it is gone when the session
ends (deleted, 60 minutes idle, or evicted when the server is full) or the
machine restarts. The audit log is off, and nothing is backed up.

## What Leaves Your Machine

### LLM providers receive

- your messages and the conversation history, cut to `llm_history_bytes`
  (131,072 bytes in the hosted profile, 524,288 otherwise);
- a dataset context block in the system prompt: dataset names and dimensions,
  column names and types, up to three sample values per column, numeric
  summaries, null shares and unique counts, cut to `llm_context_bytes` (24,576
  bytes hosted, 65,536 otherwise) and to at most 20 datasets and 60 columns per
  dataset;
- the results of the tools the model calls, each cut to at most 16 KiB within a
  tool loop.

### The hosted backend receives

every dataset you upload, your prompts and tool arguments, and your API key (in
request bodies, over TLS). It sends the key only to the provider host you chose,
which must be on its allowlist, and never stores or logs it.

### What does not communicate

- `p2a-core` makes no network requests, and DuckDB does not download extensions
  (parquet support is built in, and extension autoloading is off).
- No telemetry or usage statistics are collected.

**Interactive HTML charts** are the exception on the client side: the HTML
files they produce load plotly.js and MathJax from public CDNs
(`cdn.plot.ly`, `cdn.jsdelivr.net`) when you open them in a browser.

## Audit Log

The audit log is off unless the server runs with `P2A_AUDIT_LOG=true` (or
`--audit-log`; the variable accepts only `true` or `false`, so `1` is rejected
at startup). When it is on, it writes one JSON line per executed tool call on
every path, with `origin` `http`, `llm_tool`, `websocket` or `stdio`, and one
line for each call refused before it runs: at admission (memory pressure, a
busy session or a busy server), by a request limit or malformed arguments,
for a name no tool has (`unknown_tool`, on every path), and, for the
assistant, for an internal tool it is never offered (`not_available`). A call
refused because its session no longer exists (on any path, including a chat
whose session was removed while it ran) is not audited.

Each entry records the timestamp, `session_id`, `origin`, `tool_name`, the
sanitised `arguments`, `success`, `outcome` (`ok`, `tool_error`, `cancelled`,
`internal` for a call that panicked or failed inside the server, or a refusal
code such as `limit_exceeded`), `duration_ms` and the size of the result
(content items, text bytes, image bytes).

It does not record tool output or API keys. Argument keys that carry data
values (`content`, `csv_content`, `data`, `rows`, `values`, `filter_value`,
`base64`, ...) are replaced by their size; other strings are cut to 200 bytes,
arrays to 20 elements and nesting to 4 levels. SQL text and regex patterns are
kept, truncated like other strings, because they are what an operator needs to
investigate injected tool use. Entries still queued when the process is killed
can be lost.

## Sensitive Data Handling

### Recommendations

1. **Never commit session files** with sensitive data to version control.
2. **Keep API keys out of shared files.** Prefer environment variables on your
   own machine; never set provider keys on a public backend
   ([DEPLOYMENT.md](DEPLOYMENT.md), checklist step 1).
3. **Review audit logs before sharing them**: they contain argument values such
   as column names, file names and SQL text.
4. **Treat server logs as sensitive.** Anyone with access to them can read live
   session ids and use those sessions.
5. **Do not upload confidential data to the hosted demo.**

## MCP Tool Input Validation

The MCP server confines every file path to its data root, runs SQL read-only
with one statement per call, and, under the hosted profile, bounds the cost of
every tool call. See [PROMPT_INJECTION.md](../security/PROMPT_INJECTION.md) for
the controls and their limits when an LLM drives the analysis.

## Docker Deployment

> Hosting the web app publicly (e.g. a bring-your-own-key demo)? See
> [DEPLOYMENT.md](DEPLOYMENT.md) for the required security checklist: the
> in-process limits (`P2A_LIMITS=hosted`), no shared LLM keys, TLS, CORS
> lock-down, an edge rate limit as one layer, and container hardening.

## Data Deletion

- **Server sessions.** `DELETE /api/sessions/{id}` removes the session's
  datasets, its data directory and its database rows. Idle sessions expire
  after `P2A_SESSION_TTL` minutes (60) and are removed the same way.
- **Desktop app.** Delete the `p2a` data and configuration directories listed
  above.
- **Web app.** Clear the site data for the app's origin in your browser.
- **CLI.** Delete the session files you created (`rm -f /path/to/session.json`).
- **Docker / Compose.** `docker volume rm p2a-data p2a-db`.

## Offline Operation

- The CLI, and the desktop app with a local Ollama, need no network access.
- Interactive charts need the CDN when they are viewed.
- No telemetry, usage statistics, license checks or phone-home behaviour.
