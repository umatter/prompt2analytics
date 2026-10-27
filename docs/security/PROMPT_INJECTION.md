# Prompt Injection Security Considerations

This document describes security considerations when using prompt2analytics with LLM-driven analysis, particularly through the MCP server and Dioxus frontend.

## Overview

When an LLM (like Claude, GPT-4, or a local Ollama model) drives analytics through the MCP interface, text the model reads can steer it into tool calls the user did not intend. The server cannot tell a steered call from an intended one. What it can do is make every call, whoever asked for it, pass the same checks: confined file access, read-only SQL, bounded cost, and no reach into other sessions. This document lists those checks, which of them depend on the deployment, and what they leave open.

## Attack Vectors

### 1. Instructions embedded in data

Column names, string values or file contents can contain text that reads like an instruction to a model. The server cannot stop a model from following such text. It limits what a steered model can do:

- tools take typed, schema-checked arguments; there is no free-form code execution;
- the dataset block in the system prompt is bounded (`llm_context_bytes`, at most 20 datasets and 60 columns each), its names and sample values are JSON-quoted so data cannot start a line of the prompt, and the prompt tells the model that the block is data, not instructions;
- the model cannot call the server's internal tools (session export and import, seeds, cleaning sessions, the search meta-tools) or names no handler registers;
- every call the model makes passes the same request limits and execution governor as a direct call.

### 2. File access outside the intended directory

Every path a tool receives goes through the path jail (`crates/p2a-mcp/src/path_jail.rs`):

- paths are canonicalised (`..` and intermediate links resolved) and must stay inside the data root, `P2A_DATA_ROOT` (default: the working directory);
- relative paths resolve inside the data root, not against the process's working directory;
- a path whose final component is a symbolic link is refused, whether the link is live or dangling;
- writes go to a temporary file beside the target and are renamed into place, so a link planted at the target is replaced, never followed;
- in open public mode each session is confined to its own directory, `<data root>/sessions/<session id>/`, with a disk quota (`session_disk_bytes`, 64 MiB hosted);
- maximum file size: `load_dataset` and the dataset reload route (`POST /api/sessions/{id}/datasets/reload`) refuse a file larger than `load_file_bytes` before parsing it (64 MiB in the hosted profile, no limit when limits are off, settable in any mode with `P2A_LIMIT_OVERRIDES=load_file_bytes=<bytes>`);
- the loader chooses a parser by file extension (CSV, Parquet, Excel/ODS, Stata, SAS) and refuses any other extension;
- error messages show paths relative to the data root, never absolute server paths.

### 3. SQL

The database tools (`db_sqlite_query`, `db_duckdb_query`, `db_query_file` and the table and schema tools) apply these checks in every mode:

- exactly one statement per call: a lexer that follows DuckDB's rules for strings, quoted identifiers, dollar quotes and comments refuses any second statement;
- the first keyword must be `SELECT`, `WITH`, `EXPLAIN`, `PRAGMA`, `DESCRIBE` or `SHOW`; DuckDB statements also may not use `EXPLAIN ANALYZE` (or `EXPLAIN` with options), which executes the statement, or a `COPY` keyword;
- DuckDB database files open read-only, and every DuckDB connection has external file access, extension installation and loading, and configuration changes disabled by the engine;
- `db_query_file` reads only its jailed file: the engine may access exactly that path, and external access is then locked off;
- SQLite databases open read-only with `query_only` on, `ATTACH` disabled and only schema PRAGMAs admitted.

The hosted profile adds budgets: `db_result_rows` (100,000 rows), `db_result_bytes` (64 MiB), `db_timeout_secs` (20 s), `db_memory_mb` (256) and `db_threads` (1). A query past its time limit, or one during which the process memory grows by more than twice `db_memory_mb`, is stopped; a result cut at the row or byte budget is stored truncated and the tool output says so.

Free-form DuckDB SQL is off in the hosted profile: `sql_freeform` is `false` there, so `db_duckdb_query` and `db_query_file` refuse with "free-form SQL is not available on the public demo; load the file with load_dataset instead". An operator can re-enable them with `P2A_LIMIT_OVERRIDES=sql_freeform=true`.

With limits off (local use), the lockdown above still applies, but the size and time budgets do not, and the free-form tools run.

### 4. Resource exhaustion through tool arguments

A steered model (or any caller) can ask for very expensive work: a huge replication count, a dense method on a large dataset, a chart of absurd size. The controls:

- **request limits**: cost parameters, dataset sizes for superlinear methods and the size of a tool's output are capped; a refusal answers 422 `limit_exceeded` and names the parameter, the value requested and the limit, so a model can retry with a smaller value;
- **deadline and cancellation**: each call stops after `P2A_TOOL_TIMEOUT_SECS` (60 s hosted) and long loops stop at their next checkpoint;
- **memory admission and quotas**: a call is refused with 503 `memory_pressure` while the process RSS is above `P2A_MEMORY_SOFT_LIMIT_MB` (700 MiB hosted), a single allocation above `P2A_MAX_ALLOC_MB` (256 MiB hosted) fails as an error instead of aborting the process, and dataset bytes are capped per session (200 MiB hosted) and in total (500 MiB hosted);
- **core invariants**: in every mode, p2a-core checks parameters against the data, allocates caller-sized buffers fallibly, guards value-driven loops against NaN and infinities, and applies method-level caps (for example at most 100 outcome categories for multinomial and ordered models, tree depth at most 512 for C5.0, quantile forests and Cubist);
- `P2A_LIMITS=hosted` turns the server-side layer on for any deployment; open public mode always has it.

### 5. Server-side requests through the LLM provider URL

Callers choose the provider's `base_url`, so a steered or hostile client could point the server at an internal address. The LLM client:

- connects only to hosts on the allowlist (`P2A_LLM_ALLOWED_HOSTS`; when it is unset and `llm_allow_loopback` is false, as in the hosted profile, api.openai.com, api.anthropic.com and openrouter.ai; when it is unset and `llm_allow_loopback` is true, as in the Off profile, there is no list and any public https host is allowed);
- resolves each host itself and refuses it if any address is private, loopback, link-local or otherwise internal, then connects only to the addresses it checked;
- never follows redirects and ignores system proxies;
- never sends a server-held API key to a host the caller picked, and uses no server-held key at all while `llm_server_keys` is false (the hosted default).

### 6. Access to another session's objects

Session ids are random UUIDs and work as capabilities. Conversation, message, tool-call and settings routes check that the caller's session (the `X-P2A-Session-Id` header) owns the object and answer 404 otherwise, the same answer as for a missing object. In open public mode the routes that list sessions, browse files or count what the session store holds (`GET /api/db/stats`) do not exist, and each session's files live in its own directory.

## What the Server Enforces

Some layers are always on; others follow the resource profile, which is Hosted in open public mode and Off elsewhere unless `P2A_LIMITS=hosted` is set ([DEPLOYMENT.md](../guides/DEPLOYMENT.md), "Deployment modes").

| Control | Local | Authenticated | OpenPublic |
|---|---|---|---|
| Path jail, symlink refusal, atomic writes | on | on | on |
| Per-session data directories and disk quota | off (`P2A_SESSION_DIRS=on` enables) | off (same) | on |
| SQL lockdown (one statement, read-only engine, no file access) | on | on | on |
| SQL budgets; free-form DuckDB SQL off | with `P2A_LIMITS=hosted` | with `P2A_LIMITS=hosted` | on |
| Request limits (`limit_exceeded`) | with `P2A_LIMITS=hosted` | with `P2A_LIMITS=hosted` | on |
| Deadlines, memory admission, dataset quotas | with `P2A_LIMITS=hosted` | with `P2A_LIMITS=hosted` | on |
| Concurrency slots and panic isolation (HTTP) | on | on | on |
| Core invariants | on | on | on |
| LLM egress: internal addresses refused | on | on | on |
| LLM host allowlist | when `P2A_LLM_ALLOWED_HOSTS` is set, or `llm_allow_loopback` is false (the hosted default) | same | on unless `llm_allow_loopback=true` and `P2A_LLM_ALLOWED_HOSTS` is unset |
| Server-held LLM keys | default endpoint only, and not with the hosted profile unless `llm_server_keys=true` | same | not unless `llm_server_keys=true` (false in the hosted profile), and then default endpoint only |
| Ownership checks on conversation routes | on | on | on |
| Session-list and file-browser routes | only with `--expose-enumeration` | on | absent |
| Store-wide counts (`GET /api/db/stats`, feature `db`) | on | on | absent |

## Best Practices

### For Operators

1. **Run with minimal permissions**: use a non-root user and a dedicated, empty `P2A_DATA_ROOT`.
2. **Use the hosted profile for untrusted callers**: `P2A_LIMITS=hosted` on any server that people you do not control can reach, including one behind a shared token.
3. **Enable audit logging where you need a trail**: `P2A_AUDIT_LOG=true`; [DATA_SECURITY.md](../guides/DATA_SECURITY.md) describes where the log goes and what it records.
4. **Use network isolation or a token**: run the MCP server on localhost, or set `P2A_ACCESS_TOKEN`.
5. **Set container memory and CPU limits** as a backstop for what the in-process controls do not cover.

### For Developers Integrating p2a

1. **Validate upstream**: don't pass untrusted user input directly to tool calls
2. **Sanitize filenames**: clean user-provided filenames before using them in paths
3. **Use allowlists**: restrict which tools are available to untrusted users
4. **Monitor usage**: log and alert on unusual patterns

### For End Users

1. **Review data sources**: don't load files from untrusted sources
2. **Check results**: verify outputs make sense before acting on them
3. **Use local models**: when handling sensitive data, prefer local Ollama over cloud APIs

## Architecture Considerations

### LLM Boundary

The LLM operates in a separate context from the analytics engine:

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  User Input     │────▶│   LLM Service    │────▶│  MCP Server  │
│  (untrusted)    │     │  (interprets)    │     │  (executes)  │
└─────────────────┘     └──────────────────┘     └──────────────┘
                              │                        │
                              ▼                        ▼
                        Generates tool          Validates &
                        call requests           executes safely
```

Key principle: **The server applies the same limits regardless of whether a call came from a person, a model or an injected instruction.**

### Defense in Depth

1. **Typed arguments**: every tool call is deserialised into a typed request; unknown tools and malformed arguments are refused.
2. **Request limits**: cost parameters, dataset sizes and output size are checked before the work starts (hosted profile).
3. **Execution governor**: slots, deadlines, memory admission and dataset quotas (deadlines and quotas in the hosted profile).
4. **Core invariants**: parameter checks, fallible allocation and method-level caps in every mode.
5. **Path jail and session directories**: file access confined to the data root, and in open mode to the session's own directory.
6. **SQL lockdown**: one read-only statement, no file or extension access.
7. **Egress checks**: provider hosts allowlisted and resolved addresses vetted.
8. **Ownership checks**: a session sees only its own conversations and files.
9. **Audit trail**: when enabled, one entry per tool call that reaches the governor (executed or refused) and per assistant call of a tool it may not use (`not_available`, `unknown_tool`); DATA_SECURITY.md (Audit Log) lists what is and is not recorded.

## Known Limitations

1. **No content filtering**: the server does not scan data or prompts for injected instructions.
2. **A steered model works inside its session**: it can call any tool the model may use, read that session's data and repeat it in its answer.
3. **Trust in the LLM provider**: the provider sees your prompts and the dataset context.
4. **Local file access**: a local or desktop server can read every file under its data root, which is the working directory by default.
5. **The audit log is off by default.**
6. **stdio calls have no deadline**: the stdio transport applies request limits when a profile is configured, but runs every call to completion.

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Contact the maintainers privately
3. Provide details to reproduce the issue
4. Allow time for a fix before public disclosure

## Further Reading

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [MCP Security Best Practices](https://spec.modelcontextprotocol.io/specification/security/)
- [p2a Data Security](../guides/DATA_SECURITY.md)
