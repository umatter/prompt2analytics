# p2a-mcp HTTP API Endpoints

Comprehensive documentation of all HTTP API endpoints used by the Dioxus frontend to interact with the p2a-mcp backend.

**Base URL**: `http://localhost:8080` (configurable)
**Frontend Default Port**: `8081` (for Dioxus dev server)
**Backend Port**: `8080` (for MCP server)

Examples in this document marked "captured" are real responses of a sandboxed server, with ids shortened. How the server is deployed (local, with an access token, or as an open public server) changes which routes exist and which limits apply; [docs/guides/DEPLOYMENT.md](docs/guides/DEPLOYMENT.md) describes the modes and every configuration knob.

---

## Route index

| Method | Path | Feature | Availability | Caller's session |
|---|---|---|---|---|
| GET | `/health` | `http` | always; never needs a token | none |
| POST | `/api/sessions` | `http` | always | creates one |
| GET | `/api/sessions` | `http` | Authenticated; Local only with `--expose-enumeration`; never in OpenPublic | none |
| GET | `/api/sessions/{id}` | `http` | always | path id |
| DELETE | `/api/sessions/{id}` | `http` | always | path id |
| GET | `/api/files` | `http` | as `GET /api/sessions` | none |
| GET | `/api/tools` | `http` | always | none |
| POST | `/api/tools/{name}` | `http` | always | body `session_id` |
| GET | `/ws` | `websocket` | Local and Authenticated; OpenPublic only with `P2A_ENABLE_WS` | `auth` message |
| POST | `/api/llm/chat` | `llm` | always | body `session_id` |
| POST | `/api/llm/chat/stream` | `llm` | always | body `session_id` |
| GET | `/api/llm/models` | `llm` | always | none |
| GET | `/api/llm/env-keys` | `llm` | always | none |
| POST | `/api/llm/generate-title` | `llm` | always | none |
| POST | `/api/conversations` | `db` | with the session database | body `session_id` |
| GET | `/api/conversations` | `db` | with the session database | `X-P2A-Session-Id` |
| GET | `/api/conversations/{id}` | `db` | with the session database | `X-P2A-Session-Id` |
| PUT | `/api/conversations/{id}` | `db` | with the session database | `X-P2A-Session-Id` |
| DELETE | `/api/conversations/{id}` | `db` | with the session database | `X-P2A-Session-Id` |
| POST | `/api/conversations/{id}/archive` | `db` | with the session database | `X-P2A-Session-Id` |
| GET | `/api/conversations/{id}/messages` | `db` | with the session database | `X-P2A-Session-Id` |
| POST | `/api/conversations/{id}/messages` | `db` | with the session database | `X-P2A-Session-Id` |
| DELETE | `/api/conversations/{id}/messages` | `db` | with the session database | `X-P2A-Session-Id` |
| GET | `/api/conversations/{id}/tool-calls` | `db` | with the session database | `X-P2A-Session-Id` |
| GET | `/api/messages/{id}/tool-calls` | `db` | with the session database | `X-P2A-Session-Id` |
| GET | `/api/sessions/{session_id}/settings` | `db` | with the session database | path id |
| PUT | `/api/sessions/{session_id}/settings` | `db` | with the session database | path id |
| GET | `/api/sessions/{session_id}/datasets` | `db` | with the session database | path id |
| POST | `/api/sessions/{session_id}/datasets/reload` | `db` | with the session database | path id |
| GET | `/api/db/health` | `db` | with the session database | none |
| GET | `/api/db/stats` | `db` | with the session database | none |

- The `db` routes exist when the server is built with the `db` feature and its session database opens. The database is in memory unless `P2A_DB_PATH` names a directory.
- With `P2A_ACCESS_TOKEN` set, every route except `/health` and CORS preflight needs `Authorization: Bearer <token>`.
- A route that is not registered in the server's mode answers 404 with an empty body (`GET /api/sessions` answers 405 there, because `POST /api/sessions` exists).

---

## Session Management

### Create Session
```
POST /api/sessions
```

**Request:**
```json
{
  "user_id": null  // optional
}
```

`user_id` is optional: at most 128 bytes of `A-Z a-z 0-9 . _ @ -`, else 400 `invalid_arguments`. In open public mode it is ignored and not stored.

**Response** (201, captured):
```json
{"success":true,"data":{"session_id":"51963a14-…"}}
```

When the session table is full (`P2A_MAX_SESSIONS`), the server evicts the least recently used session that has been idle long enough (`P2A_SESSION_EVICT_IDLE_MINUTES`, 10 minutes in the hosted profile). If none qualifies, it answers (captured):

```
HTTP/1.1 503 Service Unavailable
retry-after: 60

{"success":false,"error":"Maximum number of sessions reached","code":"capacity"}
```

### Get Session Info
```
GET /api/sessions/{session_id}
```

**Response** (captured):
```json
{"success":true,"data":{"id":"51963a14-…","created_at":"2026-10-22T18:50:10.550383406Z","last_accessed":"2026-10-22T18:50:10.563349368Z","dataset_count":0,"user_id":null}}
```

An unknown session answers 404 `not_found`, an expired one 410 `session_expired`.

### List Sessions
```
GET /api/sessions
```

Registered only with an access token, or on a local server started with `--expose-enumeration`; never in open public mode (see the route index).

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "abc123def456",
      "created_at": "2026-03-07T10:30:00Z",
      "last_accessed": "2026-03-07T10:35:00Z",
      "dataset_count": 2
    }
  ]
}
```

### Delete Session
```
DELETE /api/sessions/{session_id}
```

Answers 204. It removes the session's datasets, its data directory (open public mode) and its stored conversations, messages, tool calls and settings. An unknown session answers 404.

### List Files
```
GET /api/files?path={directory}
```

A directory listing inside the server's data root. Registered as `GET /api/sessions` is; never in open public mode.

---

## Dataset Management

### List Datasets in Session
```
GET /api/sessions/{session_id}/datasets
```

The path id is the caller; it must be a live session (404 otherwise), and a `X-P2A-Session-Id` header that names another session answers 404.

**Response** (captured):
```json
{"success":true,"data":[{"id":"37962139-…","session_id":"51963a14-…","name":"sales","source_path":"sales.csv","source_type":"csv","row_count":5,"column_count":3,"column_names":["price","sqft","bedrooms"],"loaded_at":"2026-10-22T18:50:10.614694894+00:00","file_size_bytes":85}]}
```

### Reload All Datasets in Session
```
POST /api/sessions/{session_id}/datasets/reload
```

The reload runs as one governed call (memory admission, the session's slot, the deadline) inside the session's data directory, and each reloaded dataset must fit the dataset quotas. A governor refusal answers like a refused tool call (see [Errors](#errors-status-codes-and-retries)).

**Request:** (empty body)
```json
{}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "succeeded": ["sales_data", "customer_data"],
    "failed": [
      {
        "name": "archive_data",
        "source_path": "/data/archive.csv",
        "error": "File not found"
      }
    ],
    "skipped": [
      {
        "name": "temp_data",
        "reason": "Source path not set"
      }
    ]
  }
}
```

---

## Tool Management

### List Available Tools
```
GET /api/tools
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "load_dataset",
      "description": "Load a dataset from a file path (CSV, Excel, etc.)",
      "input_schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
          "session_id": { "type": "string", "description": "Session ID" },
          "path": { "type": "string", "description": "File path" },
          "name": { "type": "string", "description": "Dataset name" }
        },
        "required": ["session_id", "path"]
      }
    },
    {
      "name": "describe_dataset",
      "description": "Get summary statistics for a dataset",
      "input_schema": { ... }
    }
    // ... 268 more tools
  ]
}
```

### Call a Tool
```
POST /api/tools/{tool_name}
```

**Request:**
```json
{
  "session_id": "51963a14-…",
  "arguments": {
    "path": "sales.csv",
    "name": "sales"
  }
}
```

Paths resolve inside the server's data directory (`P2A_DATA_ROOT`), and in open public mode inside the session's own directory; a relative path is the normal form.

**Response** (captured):
```json
{"success":true,"data":{"success":true,"content":[{"type":"text","text":"Successfully loaded dataset 'sales'\n\nDimensions: 5 rows x 3 columns\n\nColumns:\n  - price (f64): 0 nulls (0.0%)\n  - sqft (i64): 0 nulls (0.0%)\n  - bedrooms (i64): 0 nulls (0.0%)"}]}}
```

A tool that runs and fails still answers HTTP 200, with `data.success: false` (captured):

```json
{"success":true,"data":{"success":false,"content":[{"type":"text","text":"Failed to load dataset: File not found: missing.csv"}],"error":"Failed to load dataset: File not found: missing.csv"}}
```

The same holds for limits a tool checks while it runs (file size, export size, join output, table cells), and for the free-form DuckDB tools in the hosted profile: there `db_duckdb_query` and `db_query_file` always answer with a tool error whose text is "free-form SQL is not available on the public demo; load the file with load_dataset instead" (`sql_freeform` is false). `db_sqlite_query` and the table and schema tools are not affected.

Every call runs under the server's execution governor. An unknown session answers 404 `"Session not found"` and an expired one 410 `"Session expired"`, both without a `code`. An unknown tool, invalid arguments, a request limit, a quota, a busy session or server, memory pressure and the deadline answer with the error contract below (400, 422, 429, 503 or 504, with `code`).

---

## LLM Chat Endpoints

Both chat routes take a slot of the session for as long as they run. In the hosted profile a session runs one chat request at a time (`llm_streams_per_session`); a second answers 429 with `code` `session_busy` and `Retry-After: 2`.

### Non-Streaming Chat
```
POST /api/llm/chat
```

**Request:**
```json
{
  "session_id": "abc123",
  "message": "What's the correlation between price and demand?",
  "history": [
    {
      "role": "user",
      "content": "Load the sales data"
    },
    {
      "role": "assistant",
      "content": "I've loaded the sales data with 1000 rows..."
    }
  ],
  "provider": {
    "provider_type": "openai",
    "api_key": "sk-...",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "interpret": true,
  "conversation_id": "conv_abc123",
  "retrieve_history": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "message": {
      "role": "assistant",
      "content": "I'll analyze the correlation...",
      "tool_calls": [
        {
          "id": "call_001",
          "name": "regression_ols",
          "arguments": {
            "dataset": "sales_data",
            "y_col": "demand",
            "x_cols": ["price"],
            "intercept": true
          }
        }
      ]
    }
  }
}
```

### Streaming Chat (SSE)
```
POST /api/llm/chat/stream
```

**Request:** (same as non-streaming)
```json
{
  "session_id": "abc123",
  "message": "What's the correlation?",
  "history": [...],
  "provider": {...},
  "interpret": true
}
```

A `conversation_id` the session does not own ends the stream with an `error` event "Conversation not found", before anything is stored or sent to the provider. Closing the SSE connection stops the tool loop.

An unknown or expired `session_id` does not change the HTTP status: the response is 200 and the stream ends with an `error` event ("Session not found" / "Session expired"). A busy session (one already running `llm_streams_per_session` chat requests; 1 in the hosted profile) is refused before the stream starts, with 429 `session_busy`.

**Response:** Server-Sent Events (SSE) stream with events:

#### Event 1: Status
```json
{
  "type": "status",
  "message": "Starting analysis..."
}
```

#### Event 2: Tool Start
```json
{
  "type": "tool_start",
  "tool": "load_dataset",
  "arguments": {
    "path": "/data/sales.csv",
    "name": "sales_data"
  }
}
```

#### Event 3: Tool End
```json
{
  "type": "tool_end",
  "tool": "load_dataset",
  "elapsed_ms": 1234,
  "result": "Successfully loaded dataset 'sales_data'"
}
```

#### Event 4: Tool Result (for visualization tools)
```json
{
  "type": "tool_result",
  "images": [
    {
      "data": "iVBORw0KGgoAAAANSUhEUgA...",
      "mime_type": "image/png"
    }
  ]
}
```

#### Event 5: Content (streaming LLM response)
```json
{
  "type": "content",
  "text": "Based on the analysis, the correlation"
}
```

```json
{
  "type": "content",
  "text": " is 0.87, indicating a strong positive relationship."
}
```

#### Event 6: Done (final message)
```json
{
  "type": "done",
  "message": {
    "role": "assistant",
    "content": "Based on the analysis, the correlation is 0.87...",
    "tool_calls": [
      {
        "id": "call_001",
        "name": "regression_ols",
        "arguments": {...}
      }
    ],
    "tool_results": [
      {
        "tool_call_id": "call_001",
        "content": "Regression results...",
        "is_error": false
      }
    ]
  }
}
```

#### Event 7: Error
```json
{
  "type": "error",
  "error": "Failed to load dataset: file not found"
}
```

**SSE Format Details:**
- Each event is sent as a line: `data: <JSON>`
- Events are separated by newlines
- Client should parse each line after `"data: "` as JSON
- Stream ends after `done` or `error` event
- Keep-alive: server sends ping every 15 seconds if no data

### List Available LLM Models
```
GET /api/llm/models
```

**Response:**
```json
{
  "success": true,
  "data": {
    "provider": "openai",
    "models": [
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-3.5-turbo"
    ]
  }
}
```

### Get LLM Environment Keys
```
GET /api/llm/env-keys
```

Whether the server would supply its own key for each provider. In the hosted profile (`llm_server_keys=false`) the server does not use its own keys, so both are `false` (captured from an open public server):

```json
{"success":true,"data":{"openai":false,"anthropic":false}}
```

### Generate Conversation Title
```
POST /api/llm/generate-title
```

**Request:**
```json
{
  "user_message": "Load the sales data and analyze it",
  "assistant_response": "I'll help you load and analyze the data...",
  "provider": {
    "provider_type": "openai",
    "model": "gpt-4o-mini"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "title": "Sales Data Analysis"
  }
}
```

---

## Conversation Management

Every `/api/conversations/{id}*` and `/api/messages/{id}/tool-calls` request names the caller's session in the `X-P2A-Session-Id: <session id>` header; the `?session_id=` query parameter is accepted as a fallback. Without either the answer is 400 `invalid_arguments` (captured):

```json
{"success":false,"error":"send the session id in the X-P2A-Session-Id header","code":"invalid_arguments"}
```

A conversation, message or tool call of another session answers 404 `not_found`, exactly as a missing one does (captured):

```json
{"success":false,"error":"Conversation not found","code":"not_found"}
```

`POST /api/conversations` takes the caller from the body's `session_id` (a header that names a different session is a 400). `/api/sessions/{session_id}/settings`, `/datasets` and `/datasets/reload` treat the path id as the caller, which must be a live session.

Limits on what a session stores (hosted values; [DEPLOYMENT.md](docs/guides/DEPLOYMENT.md) lists the others):

- a message, title or settings string longer than `message_bytes` (262,144 bytes) answers 413 `payload_too_large`;
- more than `conversations_per_session` (50) conversations or `messages_per_conversation` (1,000) messages answers 422 `quota_exceeded`;
- `session_db_bytes` (16 MiB hosted, no limit when limits are off) caps the bytes one session's history holds (titles, message contents, tool-call arguments, results and errors): a client write past it answers 422 `quota_exceeded`, text the server produces is truncated to the room left, and deleting a conversation frees its bytes;
- while the server is low on memory, writes answer 503 `memory_pressure` with `Retry-After: 15`.

Captured from an open public server:

```json
{"success":false,"error":"message is 300000 bytes; this server accepts at most 262144 bytes per message","code":"payload_too_large"}
```

```json
{"success":false,"error":"this session already has 50 conversations, the most this server keeps; delete one first","code":"quota_exceeded"}
```

### List Conversations
```
GET /api/conversations?session_id={session_id}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "conv_001",
      "session_id": "abc123",
      "title": "Sales Analysis",
      "created_at": "2026-03-07T10:00:00Z",
      "updated_at": "2026-03-07T10:30:00Z",
      "is_archived": false,
      "message_count": 5,
      "last_message_preview": "Based on the analysis, correlation is..."
    }
  ]
}
```

### Create Conversation
```
POST /api/conversations
```

**Request:**
```json
{
  "session_id": "abc123",
  "title": "New Analysis"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "conv_new",
    "session_id": "abc123",
    "title": "New Analysis",
    "created_at": "2026-03-07T11:00:00Z",
    "updated_at": "2026-03-07T11:00:00Z",
    "is_archived": false,
    "message_count": 0
  }
}
```

### Get Conversation
```
GET /api/conversations/{conversation_id}
```

**Response** (captured): the conversation and its messages.
```json
{"success":true,"data":{"conversation":{"id":"e104f3c9-…","session_id":"51963a14-…","title":"Sales analysis","created_at":"2026-10-22T18:50:10.649869292+00:00","updated_at":"2026-10-22T18:50:10.673463679+00:00","is_archived":false,"message_count":1,"last_message_preview":"Regress price on sqft"},"messages":[{"id":"4ea7256f-…","conversation_id":"e104f3c9-…","role":"user","content":"Regress price on sqft","created_at":"2026-10-22T18:50:10.672294916+00:00"}]}}
```

### Get Conversation with Messages
```
GET /api/conversations/{conversation_id}/full
```

Not implemented by the server in this version.

**Response:**
```json
{
  "success": true,
  "data": {
    "conversation": {
      "id": "conv_001",
      "session_id": "abc123",
      "title": "Sales Analysis",
      ...
    },
    "messages": [
      {
        "id": "msg_001",
        "conversation_id": "conv_001",
        "role": "user",
        "content": "Load sales data",
        "created_at": "2026-03-07T10:00:00Z",
        "token_count": 5,
        "model": "gpt-4o-mini"
      },
      {
        "id": "msg_002",
        "conversation_id": "conv_001",
        "role": "assistant",
        "content": "I'll load the sales data...",
        "created_at": "2026-03-07T10:01:00Z",
        "token_count": 42,
        "model": "gpt-4o-mini"
      }
    ]
  }
}
```

### Update Conversation
```
PUT /api/conversations/{conversation_id}
```

**Request:**
```json
{
  "title": "Updated Title",
  "is_archived": false
}
```

### Archive Conversation
```
POST /api/conversations/{conversation_id}/archive
```

**Request:**
```json
{"is_archived":true}
```

**Response** (captured): the updated conversation.
```json
{"success":true,"data":{"id":"e104f3c9-…","session_id":"51963a14-…","title":"Sales analysis","created_at":"2026-10-22T18:50:10.649869292+00:00","updated_at":"2026-10-22T18:50:10.702766403+00:00","is_archived":true,"message_count":1,"last_message_preview":"Regress price on sqft"}}
```

### Get Messages
```
GET /api/conversations/{conversation_id}/messages
```

### Add Message
```
POST /api/conversations/{conversation_id}/messages
```

**Request:**
```json
{
  "role": "user",
  "content": "Continue the analysis"
}
```

`role` is `user`, `assistant` or `system` (400 `invalid_arguments` otherwise).

**Response** (201, captured):
```json
{"success":true,"data":{"id":"4ea7256f-…","conversation_id":"e104f3c9-…","role":"user","content":"Regress price on sqft","created_at":"2026-10-22T18:50:10.672294916+00:00"}}
```

### Delete Messages
```
DELETE /api/conversations/{conversation_id}/messages
```

**Response** (captured):
```json
{"success":true,"data":{"deleted_count":1}}
```

### Clear Messages
```
POST /api/conversations/{conversation_id}/messages/clear
```

Not implemented by the server in this version.

**Request:** (empty body)
```json
{}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "deleted_count": 10
  }
}
```

### Get Tool Calls of a Conversation
```
GET /api/conversations/{conversation_id}/tool-calls
```

### Get Tool Calls of a Message
```
GET /api/messages/{message_id}/tool-calls
```

Both answer a list of the stored tool calls (captured, for a conversation without any):

```json
{"success":true,"data":[]}
```

### Delete Conversation
```
DELETE /api/conversations/{conversation_id}
```

Answers 204.

### Session Settings
```
GET /api/sessions/{session_id}/settings
PUT /api/sessions/{session_id}/settings
```

The path id is the caller. `PUT` takes any of `provider`, `model`, `temperature` and `max_tokens`. **Response** (captured, after a `PUT` with `{"provider":"openai","model":"gpt-4o-mini"}`):

```json
{"success":true,"data":{"id":{"tb":"settings","id":{"String":"51963a14-…"}},"session_id":"51963a14-…","provider":"openai","model":"gpt-4o-mini","base_url":null,"temperature":0.7,"max_tokens":4096,"system_prompt":null,"updated_at":"2026-10-22T18:50:10.735301052Z"}}
```

### Database Health and Statistics
```
GET /api/db/health
GET /api/db/stats
```

**Responses** (captured):
```json
{"success":true,"data":{"healthy":true}}
```

```json
{"success":true,"data":{"session_count":1,"conversation_count":1,"message_count":1}}
```

---

## WebSocket (`/ws`)

```
GET /ws
```

A websocket for tool calls. It exists only when the server is built with the `websocket` feature, and in open public mode only with `P2A_ENABLE_WS` (`--enable-ws`). Messages are JSON objects with a `type` field.

| Direction | `type` | Fields |
|---|---|---|
| client → server | `auth` | `session_id` |
| client → server | `tool` | `id`, `name`, `args` |
| client → server | `chat` | `id`, `message`, `stream` (not implemented; answers an `error`) |
| client → server | `ping` | |
| server → client | `auth_result` | `success`, `error` (optional) |
| server → client | `tool_result` | `id`, `success`, `result` (the tool result, as in `POST /api/tools/{name}`) |
| server → client | `done` | `id` |
| server → client | `error` | `id`, `message`, `code` (optional) |
| server → client | `pong` | |

A connection authenticates once with `auth`; each `tool` message then runs under the same governor and request limits as `POST /api/tools/{name}`, and the session is looked up again for every call. An `error` carries a `code` when a tool message is refused: a governor or limit code from the table below (`unknown_tool`, `limit_exceeded`, `session_busy`, ...), or `session_not_found` / `session_expired` when the authenticated session has gone. Messages and frames are capped at the HTTP body cap (`P2A_MAX_HTTP_BODY_MB`). An edge rate limiter counts the whole connection as one request.

Captured exchange (`>` sent, `<` received):

```
> {"type": "tool", "id": "t0", "name": "list_datasets", "args": {}}
< {"type":"error","id":"t0","message":"Not authenticated. Send auth message first."}
> {"type": "auth", "session_id": "5dbee6be-…"}
< {"type":"auth_result","success":true}
> {"type": "ping"}
< {"type":"pong"}
> {"type": "tool", "id": "t1", "name": "no_such_tool", "args": {}}
< {"type":"error","id":"t1","message":"Unknown tool: no_such_tool","code":"unknown_tool"}
> {"type": "tool", "id": "t2", "name": "list_datasets", "args": {}}
< {"type":"tool_result","id":"t2","success":true,"result":{"success":true,"content":[{"type":"text","text":"No datasets currently loaded. Use the 'load_dataset' tool to load a data file."}]}}
< {"type":"done","id":"t2"}
```

---

## Health & System

### Health Check
```
GET /health
```

**Response** (captured, local server):
```json
{
  "status": "ok",
  "server": "p2a-mcp",
  "version": "0.1.3",
  "active_sessions": 0,
  "governor": {
    "in_flight": 0,
    "overdue": 0,
    "capacity": 12,
    "rss_bytes": 79233024,
    "healthy": true
  }
}
```

The `governor` object reports the tool calls running (`in_flight`), those past their deadline (`overdue`), the number of tool slots (`capacity`, `P2A_TOOL_CONCURRENCY`) and the process RSS. In open public mode it leaves out `in_flight` and reports `rss_bytes` as `null` (captured, open public server):

```json
{
  "status": "ok",
  "server": "p2a-mcp",
  "version": "0.1.3",
  "active_sessions": 0,
  "governor": {
    "healthy": true,
    "overdue": 0,
    "capacity": 12,
    "rss_bytes": null
  }
}
```

While every slot is held by work past its deadline, `/health` answers 503 with `"status": "degraded"` and `"healthy": false`. In the hosted profile the process exits (code 70) when that lasts `P2A_STUCK_EXIT_SECS` (300 s), so the platform can restart it.

---

## Request/Response Types Reference

### Message (History Item)
```typescript
interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  tool_calls?: ToolCall[];      // For assistant messages
  tool_results?: ToolResult[];   // For tool role messages
}
```

### ToolCall
```typescript
interface ToolCall {
  id: string;                    // Unique call ID
  name: string;                  // Tool name (e.g., "regression_ols")
  arguments: Record<string, any>; // Tool arguments
}
```

### ToolResult
```typescript
interface ToolResult {
  tool_call_id: string;          // References ToolCall.id
  content: string;               // Tool output/result
  is_error: boolean;             // Whether tool call failed
}
```

### ProviderConfig
```typescript
interface ProviderConfig {
  provider_type: "openai" | "anthropic" | "ollama";
  api_key?: string;              // Required unless the server may use its own key for this provider:
                                 // never in the hosted profile, and only when base_url is unset or
                                 // the provider's default endpoint
  base_url?: string;             // Custom base URL; must pass the server's LLM host allowlist
  model: string;                 // Model name (e.g., "gpt-4o-mini")
  temperature?: number;          // 0.0-2.0 (default: varies by provider)
  max_tokens?: number;           // Maximum output tokens
}
```

### LlmChatRequest
```typescript
interface LlmChatRequest {
  session_id: string;            // Session identifier
  message: string;               // User message
  provider?: ProviderConfig;     // LLM provider (uses env defaults if not provided)
  history?: Message[];           // Conversation history
  interpret: boolean;            // Whether to auto-execute tools (default: true)
  conversation_id?: string;      // For persistent conversations (optional)
  retrieve_history: boolean;     // Auto-retrieve history from DB (default: true)
  max_tool_iterations?: number;  // Tool-loop rounds (default 25), clamped to the server's
                                 // llm_max_tool_iterations (25 hosted, 50 otherwise)
}
```

### StreamEvent (in SSE stream)
```typescript
type StreamEvent =
  | { type: "status"; message: string }
  | { type: "tool_start"; tool: string; arguments: Record<string, any> }
  | { type: "tool_end"; tool: string; elapsed_ms: number; result?: string }
  | { type: "tool_result"; images?: { data: string; mime_type: string }[] }
  | { type: "content"; text: string }
  | { type: "done"; message: Message }
  | { type: "error"; error: string };
```

---

## Errors, status codes and retries

An error response has this shape; `data` is absent, and `code` and `details` appear where the table below lists them:

```json
{"success": false, "error": "<plain-language message>", "code": "<code>", "details": {}}
```

| Status | `code` | Meaning |
|---|---|---|
| 400 | `unknown_tool` | no tool has this name |
| 400 | `invalid_arguments` | malformed tool arguments, `user_id`, message role, or a missing or conflicting session id on the conversation routes |
| 401 | none | a token is configured and the request has none or a wrong one: plain-text body `missing or invalid bearer token` and `WWW-Authenticate: Bearer` |
| 404 | `not_found` | missing, or not owned by the caller's session. A route not registered in this mode answers 404 with an empty body; an unknown session on `POST /api/tools/{name}` or `/api/llm/chat` answers 404 `"Session not found"` without a `code` (the stream route reports these as an SSE `error` event) |
| 410 | `session_expired` | the session expired (session and conversation routes); on `POST /api/tools/{name}` and `/api/llm/chat` an expired session answers 410 `"Session expired"` without a `code` (the stream route reports these as an SSE `error` event) |
| 413 | `payload_too_large` | a message, title or settings string above `message_bytes`. A request body above `P2A_MAX_HTTP_BODY_MB` answers 413 with the plain-text body `length limit exceeded` and no `code` |
| 421 | `misdirected` | a local server received a request addressed to a host other than `localhost` or a loopback address |
| 422 | `limit_exceeded` | a request limit refused a parameter, the dataset's row count or the response size; `details {parameter, requested, limit}` |
| 422 | `quota_exceeded` | a dataset quota (the call's changes were undone) or a conversation-store bound |
| 429 | `session_busy` | the session's slot is taken: `Retry-After: 2` |
| 503 | `server_busy` | every tool slot is taken: `Retry-After: 5` |
| 503 | `memory_pressure` | the process is above its memory soft limit: `Retry-After: 15` |
| 503 | `capacity` | the session table is full: `Retry-After: 60` |
| 504 | `deadline_exceeded` | the tool call ran past the server's time limit; `details {limit_secs}` |
| 500 | `internal` | the tool panicked or the server failed; the session survives |

The non-streaming `/api/llm/chat` also answers 400 for a refused provider configuration (host not allowed, missing key), 502 when the provider returned an error and 504 when it did not answer in time, without a `code`.

A limit refusal (captured, open public server):

```
HTTP/1.1 422 Unprocessable Entity
content-type: application/json

{"success":false,"error":"Parameter `n_boot` = 1000000 exceeds this server's limit of 10000 (replications) for `regression_bootstrap_cov`. Use a smaller value.","code":"limit_exceeded","details":{"parameter":"n_boot","requested":"1000000","limit":"10000"}}
```

- Clients should honour `Retry-After`; CORS exposes it to browser clients.
- In the LLM tool loop these refusals reach the model as the tool's error text (`is_error: true`), so it can retry with a smaller request.
- On the stdio transport a limit refusal is a JSON-RPC invalid-params error (-32602) whose `data.code` is `"limit_exceeded"`, with `parameter`, `requested` and `limit`.

**Successful status codes:**
- `200 OK` - Successful request
- `201 Created` - Resource created
- `204 No Content` - Session or conversation deleted

---

## Data Loading via Tools

Datasets are loaded using the `load_dataset` tool (not a direct HTTP endpoint):

**Tool: load_dataset**
```json
{
  "session_id": "abc123",
  "path": "data.csv",
  "name": "my_data"
}
```

The path resolves inside the server's data directory (per session in open public mode). In the hosted profile a file larger than `load_file_bytes` (64 MiB) is refused before it is parsed.

Supported formats:
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- Parquet (`.parquet`)
- JSON Lines (`.jsonl`)
- Stata (`.dta`)
- SAS (`.sas7bdat`)

After loading, datasets are stored in the session and can be referenced by name in subsequent tool calls.

---

## Streaming Implementation Details

### Web (WASM) Implementation
- Uses Fetch API with streaming response body
- ReadableStream + BYOB reader for efficient memory usage
- Chunks accumulated in buffer and split by newlines
- Each `data: <JSON>` line parsed as StreamEvent

### Native Implementation
- Uses `reqwest` with streaming response
- Chunks accumulated in buffer, split by newlines
- SSE parser handles `data:` prefix and JSON deserialization

### SSE Parser
```rust
fn parse_sse_line(line: &str) -> Option<StreamEvent> {
    let line = line.trim();

    // Skip empty lines and comments
    if line.is_empty() || line.starts_with(':') {
        return None;
    }

    // Parse data lines
    if let Some(data) = line.strip_prefix("data: ") {
        serde_json::from_str::<StreamEvent>(data).ok()
    } else {
        None
    }
}
```

---

## Integration Notes for Evaluation

The paper's evaluation harness drives a loopback server without a token, which runs in local mode with no resource limits, so the endpoints it uses are unchanged: `POST /api/sessions`, `POST /api/tools/load_dataset`, `POST /api/llm/chat/stream` and `GET /health`. Responses only gain the optional `code` and `details` fields on errors and the `governor` object in `/health`. Against a server with `P2A_LIMITS=hosted`, tool calls can also answer 422, 429, 503 or 504.

### For Automated Testing
1. **Create session first**: `POST /api/sessions` → get `session_id`
2. **Prepare datasets**: Use `load_dataset` tool via `/api/llm/chat` or directly
3. **Stream chat**: `POST /api/llm/chat/stream` with `retrieve_history: true`
4. **Parse SSE events**: Listen for `tool_start`, `tool_end`, `content`, `done` events
5. **Extract results**: From `tool_end.result` or final message content

### Key Points for Scripting
- **Conversation ID** is optional but enables history persistence
- **`retrieve_history: true`** (default) automatically loads prior turns from database
- **Tool calls** in assistant message are paired with tool results via matching `tool_call_id`
- **`is_error: false`** field is required in `ToolResult` for OpenAI compatibility
- **SSE stream ends** after `done` or `error` event

### Dataset Context
When making LLM requests, the backend automatically provides enhanced context about loaded datasets:
- Column names and data types
- Sample values (first 3 rows)
- Numeric ranges and statistics
- Null percentages
- Unique value counts for categoricals

This context is included in the system prompt to help the LLM make better decisions about which tools to call. It is bounded: at most `llm_context_bytes` (24,576 bytes in the hosted profile, 65,536 otherwise), 20 datasets and 60 columns per dataset.
