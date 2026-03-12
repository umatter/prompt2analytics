# p2a-mcp HTTP API Endpoints

Comprehensive documentation of all HTTP API endpoints used by the Dioxus frontend to interact with the p2a-mcp backend.

**Base URL**: `http://localhost:8080` (configurable)
**Frontend Default Port**: `8081` (for Dioxus dev server)
**Backend Port**: `8080` (for MCP server)

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

**Response:**
```json
{
  "success": true,
  "data": {
    "session_id": "abc123def456"
  }
}
```

### Get Session Info
```
GET /api/sessions/{session_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "abc123def456",
    "created_at": "2026-03-07T10:30:00Z",
    "last_accessed": "2026-03-07T10:35:00Z",
    "dataset_count": 2,
    "user_id": null
  }
}
```

### List Sessions
```
GET /api/sessions
```

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

---

## Dataset Management

### List Datasets in Session
```
GET /api/sessions/{session_id}/datasets
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "dataset_1",
      "session_id": "abc123",
      "name": "sales_data",
      "source_path": "/data/sales.csv",
      "source_type": "csv",
      "row_count": 1000,
      "column_count": 8,
      "column_names": ["date", "product", "amount", "region", ...],
      "loaded_at": "2026-03-07T10:30:00Z",
      "file_size_bytes": 54321
    }
  ]
}
```

### Reload All Datasets in Session
```
POST /api/sessions/{session_id}/datasets/reload
```

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
  "session_id": "abc123",
  "arguments": {
    "path": "/path/to/data.csv",
    "name": "my_dataset"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "success": true,
    "content": [
      {
        "type": "text",
        "text": "Successfully loaded dataset 'my_dataset' with 1000 rows and 8 columns"
      }
    ]
  }
}
```

---

## LLM Chat Endpoints

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

**Response:**
```json
{
  "success": true,
  "data": {
    "openai": true,
    "anthropic": false
  }
}
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

### Get Conversation with Messages
```
GET /api/conversations/{conversation_id}/full
```

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

### Clear Messages
```
POST /api/conversations/{conversation_id}/messages/clear
```

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

### Delete Conversation
```
DELETE /api/conversations/{conversation_id}
```

---

## Health & System

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "server": "p2a-mcp",
  "version": "0.4.0",
  "active_sessions": 3
}
```

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
  api_key?: string;              // API key (optional, uses env vars if not provided)
  base_url?: string;             // Custom base URL (for Ollama, etc.)
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

## Error Responses

All error responses follow this format:

```json
{
  "success": false,
  "error": "Descriptive error message",
  "data": null
}
```

**HTTP Status Codes:**
- `200 OK` - Successful request
- `201 Created` - Resource created
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - LLM provider not available

---

## Data Loading via Tools

Datasets are loaded using the `load_dataset` tool (not a direct HTTP endpoint):

**Tool: load_dataset**
```json
{
  "session_id": "abc123",
  "path": "/path/to/data.csv",
  "name": "my_data"
}
```

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

This context is included in the system prompt to help the LLM make better decisions about which tools to call.
