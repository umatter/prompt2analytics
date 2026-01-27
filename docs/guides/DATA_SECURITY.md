# Data Security and Privacy

This document describes where `prompt2analytics` stores data, how it handles user information, and privacy considerations.

## Overview

**prompt2analytics operates entirely locally** - no data is transmitted to external servers by the analytics engine itself. All computations happen on your machine.

## Data Storage Locations

### CLI (`p2a`)

| Data Type | Location | Purpose |
|-----------|----------|---------|
| Session files | User-specified via `--session` | Recording commands for reproducibility |
| Datasets | In-memory only | Not persisted between runs (unless session enabled) |
| Output files | User-specified paths | Charts, exports, saved datasets |

The CLI does not create any files in hidden directories or system locations unless explicitly requested.

### MCP Server (`p2a-mcp`)

| Data Type | Location | Purpose |
|-----------|----------|---------|
| Audit log | `~/.p2a/audit.log` | Operation history for debugging |
| Session database | `~/.p2a/surrealdb/` | SurrealDB embedded database (RocksDB backend) |
| Datasets | In-memory per session | Temporary storage during analysis |
| Uploaded files | Processed in-memory | Not persisted after session ends |

To clear all MCP server data:
```bash
rm -rf ~/.p2a/
```

### Dioxus App (`p2a-dioxus`)

| Data Type | Location | Purpose |
|-----------|----------|---------|
| Conversation history | Backend database (`~/.p2a/surrealdb/`) | Chat message persistence |
| LLM settings | Browser localStorage (web) or app state (desktop) | Provider configuration |
| API keys | Environment variables only | Never stored in files |

## Network Communication

### What DOES communicate over the network:

1. **LLM providers** (when configured): Dioxus app sends prompts to:
   - Ollama (typically `localhost:11434`)
   - Anthropic API (`api.anthropic.com`)
   - OpenAI API (`api.openai.com`)

2. **MCP server HTTP mode**: When run with `--transport http`, listens on specified port (default 8080)

### What does NOT communicate:

- The analytics engine (p2a-core) makes no network requests
- Dataset contents are never sent externally by the analytics code
- Results and computations stay local

## Sensitive Data Handling

### Recommendations

1. **Never commit session files** with sensitive data to version control
2. **Clear `~/.p2a/` directory** when switching between projects
3. **Use environment variables** for API keys, not config files
4. **Review audit logs** before sharing for debugging

### Audit Log Contents

The audit log (`~/.p2a/audit.log`) contains:
- Timestamps of operations
- Tool names invoked
- Dataset names (not contents)
- Error messages

It does NOT contain:
- Actual data values
- Full query results
- API keys or credentials

## MCP Tool Input Validation

The MCP server validates all tool inputs to prevent:
- Path traversal attacks
- SQL injection (when using database tools)
- Arbitrary code execution

See [PROMPT_INJECTION.md](../security/PROMPT_INJECTION.md) for security considerations when using LLM-driven analysis.

## Docker Deployment

When running via Docker:

| Data Type | Location | Notes |
|-----------|----------|-------|
| Database | Docker volume `p2a-data` | Persists across container restarts |
| Audit log | Container filesystem | Lost on container removal unless volume-mounted |

For production deployments, mount a volume for persistence:
```bash
docker run -v p2a-data:/root/.p2a p2a-mcp
```

## Data Deletion

To completely remove all prompt2analytics data:

```bash
# Remove MCP server data
rm -rf ~/.p2a/

# Remove any session files (user-specified locations)
rm -f /path/to/your/session.json

# Docker: Remove volumes
docker volume rm p2a-data
```

## Offline Operation

All analytics functions work completely offline:
- No internet connection required for computations
- No telemetry or usage statistics collected
- No license validation or phone-home behavior

The only features requiring network access are:
- LLM integration (optional, can use local Ollama)
- Package updates via `cargo update`
