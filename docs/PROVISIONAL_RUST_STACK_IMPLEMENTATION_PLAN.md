# Full Rust Stack Implementation Plan

## Status: VALIDATED (Server-Side Only)

This document outlines the migration path to a unified full Rust stack using **Dioxus** (frontend) and **SurrealDB** (database). It builds upon the earlier [DIOXUS_MIGRATION_FEASIBILITY.md](./DIOXUS_MIGRATION_FEASIBILITY.md) study and incorporates learnings from the working `p2a-dioxus` prototype.

**Update (Jan 2026):** After prototyping SurrealDB with WASM/IndexedDB, we determined that **server-side persistence only** is the recommended approach. See [WASM/IndexedDB Investigation](#wasmindexdb-investigation-results) for details.

---

## Executive Summary

### Goals

1. **Unified Rust Stack**: Single language (Rust) for frontend, backend, and persistence
2. **Persistent Conversations**: Store chat history, messages, and tool calls across sessions
3. **Cross-Platform Deployment**: Same Dioxus codebase targets web (WASM) and desktop
4. **Eliminate Node.js Dependencies**: No more npm, TypeScript, or JavaScript toolchain

### Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                     Dioxus WASM Frontend                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Dioxus Components (RSX)                     │   │
│  │   ChatPanel | MessageList | ConversationSidebar | Settings   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              localStorage (settings cache only)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                                 │ HTTP/SSE
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                        p2a-mcp Backend                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ HTTP/SSE API │  │ SessionMgr   │  │ 55+ MCP Analytics Tools  │  │
│  │ (axum)       │  │ (per-user)   │  │ (via p2a-core)           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│                           │                                         │
│                           ▼                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   SurrealDB Layer                            │   │
│  │   conversations | messages | tool_calls | sessions | settings│   │
│  │   (Embedded RocksDB for persistence)                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                        p2a-core                                     │
│   Pure Rust: OLS, Panel, IV, DiD, Discrete, TimeSeries, ML, Viz    │
│   (32K+ lines, unchanged)                                           │
└────────────────────────────────────────────────────────────────────┘
```

---

## WASM/IndexedDB Investigation Results

### Prototype Attempt

We attempted to integrate SurrealDB with `kv-indxdb` (IndexedDB backend) directly in the Dioxus WASM frontend.

### Build Failure

```
error: failed to run custom build command for `ring v0.17.14`
error occurred in cc-rs: failed to find tool "clang"
```

The `ring` crate (SurrealDB dependency for cryptography) requires native C compilation, which doesn't work for `wasm32-unknown-unknown` targets without special tooling.

### Known Open Issues (as of Jan 2026)

| Issue | Status | Impact |
|-------|--------|--------|
| [#5712 - Transaction overlapping](https://github.com/surrealdb/surrealdb/issues/5712) | **Open** | Chrome hangs, Firefox throws errors under concurrent load |
| [#5712 - Memory leaks](https://github.com/surrealdb/surrealdb/issues/5712) | **Open** | ~2MB per connection, cumulative |
| [#6311 - Changefeed errors](https://github.com/surrealdb/surrealdb/issues/6311) | **Open** (Sept 2025) | Periodic TransactionInactiveError in Dioxus apps |

### Root Cause

The JavaScript SDK implements transaction queuing to work around IndexedDB limitations, but the Rust SDK doesn't. This creates an "async-barrier" where transactions complete on Rust's side before JavaScript promises resolve, breaking IndexedDB semantics.

### Decision

**Server-side SurrealDB only.** The frontend will use HTTP API for all persistence, with localStorage for caching settings only.

---

## Current vs Target Architecture

### Current State

| Component | Technology | Persistence |
|-----------|------------|-------------|
| p2a-mcp | Rust, axum HTTP, 55+ tools | SurrealDB (conversations, datasets) |
| p2a-dioxus | Rust, Dioxus 0.7, WASM/Native | localStorage (settings) |
| p2a-core | Pure Rust, ndarray, faer | N/A (analytics library) |
| p2a-cli | Rust, clap | Session JSON files |

**Current Session State (p2a-mcp):**
- `SessionManager` with `Arc<RwLock<HashMap<SessionId, Arc<Session>>>>`
- Sessions contain: `datasets` (HashMap), `global_seed`, `user_id`, timestamps
- **Volatile**: All data lost on server restart
- Background cleanup task for expired sessions

### Target State

| Component | Technology | Persistence |
|-----------|------------|-------------|
| p2a-dioxus | Dioxus 0.7, WASM + Desktop | Via HTTP API (server-side) |
| p2a-mcp | Rust, axum HTTP | SurrealDB (RocksDB embedded) |
| p2a-core | Pure Rust (unchanged) | N/A |

**Target Session State:**
- Sessions persist across server restarts
- Conversation history with full message/tool_call records
- User settings and preferences stored server-side
- Datasets remain in-memory (Polars DataFrames not serializable to DB)

---

## SurrealDB Integration Strategy

### Why SurrealDB?

1. **Native Rust SDK**: `cargo add surrealdb`, no FFI or bindings
2. **Embedded Mode**: `connect("rocksdb://./data")` - no separate server process
3. **LIVE Queries**: Real-time subscriptions for collaborative features (future)
4. **SurrealQL**: SQL-like syntax, easy learning curve
5. **Flexible Schema**: Schemaless or strict, adapts as needs evolve

### Schema Design

```surql
-- Conversations table
DEFINE TABLE conversations SCHEMAFULL;
DEFINE FIELD id ON conversations TYPE string;
DEFINE FIELD session_id ON conversations TYPE string;
DEFINE FIELD title ON conversations TYPE string;
DEFINE FIELD created_at ON conversations TYPE datetime;
DEFINE FIELD updated_at ON conversations TYPE datetime;
DEFINE FIELD is_archived ON conversations TYPE bool DEFAULT false;
DEFINE INDEX idx_conversations_session ON conversations COLUMNS session_id;
DEFINE INDEX idx_conversations_updated ON conversations COLUMNS updated_at;

-- Messages table
DEFINE TABLE messages SCHEMAFULL;
DEFINE FIELD id ON messages TYPE string;
DEFINE FIELD conversation_id ON messages TYPE string;
DEFINE FIELD role ON messages TYPE string ASSERT $value IN ['user', 'assistant', 'system'];
DEFINE FIELD content ON messages TYPE string;
DEFINE FIELD created_at ON messages TYPE datetime;
DEFINE FIELD token_count ON messages TYPE option<int>;
DEFINE INDEX idx_messages_conversation ON messages COLUMNS conversation_id;
DEFINE INDEX idx_messages_created ON messages COLUMNS created_at;

-- Tool calls table (linked to messages)
DEFINE TABLE tool_calls SCHEMAFULL;
DEFINE FIELD id ON tool_calls TYPE string;
DEFINE FIELD message_id ON messages TYPE string;
DEFINE FIELD tool_name ON tool_calls TYPE string;
DEFINE FIELD arguments ON tool_calls TYPE string;  -- JSON string
DEFINE FIELD result ON tool_calls TYPE option<string>;  -- JSON string
DEFINE FIELD status ON tool_calls TYPE string ASSERT $value IN ['pending', 'success', 'error'];
DEFINE FIELD started_at ON tool_calls TYPE datetime;
DEFINE FIELD completed_at ON tool_calls TYPE option<datetime>;
DEFINE FIELD duration_ms ON tool_calls TYPE option<int>;
DEFINE INDEX idx_tool_calls_message ON tool_calls COLUMNS message_id;

-- Sessions table (extends current in-memory sessions)
DEFINE TABLE sessions SCHEMAFULL;
DEFINE FIELD id ON sessions TYPE string;
DEFINE FIELD user_id ON sessions TYPE option<string>;
DEFINE FIELD created_at ON sessions TYPE datetime;
DEFINE FIELD last_accessed ON sessions TYPE datetime;
DEFINE FIELD global_seed ON sessions TYPE option<int>;
DEFINE FIELD metadata ON sessions TYPE option<object>;
DEFINE INDEX idx_sessions_user ON sessions COLUMNS user_id;

-- User settings table
DEFINE TABLE settings SCHEMAFULL;
DEFINE FIELD id ON settings TYPE string;
DEFINE FIELD session_id ON settings TYPE string;
DEFINE FIELD provider ON settings TYPE string;
DEFINE FIELD model ON settings TYPE string;
DEFINE FIELD api_key ON settings TYPE option<string>;  -- Encrypted in practice
DEFINE FIELD base_url ON settings TYPE option<string>;
DEFINE FIELD temperature ON settings TYPE option<float>;
DEFINE FIELD max_tokens ON settings TYPE option<int>;
DEFINE FIELD updated_at ON settings TYPE datetime;
DEFINE INDEX idx_settings_session ON settings COLUMNS session_id UNIQUE;
```

### Datasets: In-Memory Only

**Important Design Decision**: Polars `DataFrame` objects cannot be serialized to SurrealDB. The current approach (datasets in HashMap) will be retained:

- Datasets loaded via `load_csv`, `load_parquet` etc. remain in-memory
- Session metadata (which datasets are loaded, their names) can be persisted
- On restart, users must re-load datasets
- Future enhancement: persist dataset file paths for "reload last session" feature

---

## Implementation Phases

### Phase 1: Backend Database Layer (~5 days)

#### 1.1 Add SurrealDB Dependency

```toml
# crates/p2a-mcp/Cargo.toml
[dependencies]
surrealdb = { version = "2.4", features = ["kv-rocksdb"] }
```

#### 1.2 Create Database Module

New file structure in `p2a-mcp`:
```
src/
├── db/
│   ├── mod.rs           # Module exports
│   ├── connection.rs    # SurrealDB connection management
│   ├── models.rs        # Rust structs for DB records
│   ├── conversations.rs # Conversation CRUD
│   ├── messages.rs      # Message CRUD
│   └── settings.rs      # Settings CRUD
├── session.rs           # Extend with DB persistence
└── server.rs            # Add conversation endpoints
```

#### 1.3 Database Connection Manager

```rust
// src/db/connection.rs
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use std::sync::Arc;

pub struct DbConnection {
    db: Arc<Surreal<RocksDb>>,
}

impl DbConnection {
    pub async fn connect(path: &str) -> Result<Self, surrealdb::Error> {
        let db = Surreal::new::<RocksDb>(path).await?;
        db.use_ns("p2a").use_db("analytics").await?;

        // Run migrations/schema setup
        Self::setup_schema(&db).await?;

        Ok(Self { db: Arc::new(db) })
    }

    async fn setup_schema(db: &Surreal<RocksDb>) -> Result<(), surrealdb::Error> {
        // Execute schema definitions
        db.query(include_str!("../schema.surql")).await?;
        Ok(())
    }
}
```

#### 1.4 Extend SessionManager

```rust
// Modified session.rs
impl SessionManager {
    /// Create a new session and persist to database.
    pub async fn create_session(&self, user_id: Option<String>) -> Result<SessionId, SessionError> {
        // ... existing logic ...

        // Persist to SurrealDB
        if let Some(db) = &self.db {
            db.create_session(&session).await?;
        }

        Ok(id)
    }

    /// Load session from database if not in memory.
    pub async fn get_session(&self, id: &str) -> Result<Arc<Session>, SessionError> {
        // Check memory first
        if let Some(session) = self.sessions.read().await.get(id) {
            return Ok(session.clone());
        }

        // Try loading from database
        if let Some(db) = &self.db {
            if let Some(session) = db.load_session(id).await? {
                // Add to memory cache
                self.sessions.write().await.insert(id.to_string(), session.clone());
                return Ok(session);
            }
        }

        Err(SessionError::NotFound)
    }
}
```

#### 1.5 New API Endpoints

Add to `server.rs`:

```rust
// GET /api/conversations - List conversations for session
// POST /api/conversations - Create new conversation
// GET /api/conversations/{id} - Get conversation with messages
// PUT /api/conversations/{id} - Update conversation (title, archive)
// DELETE /api/conversations/{id} - Delete conversation

// Messages are typically fetched with conversation
// POST /api/conversations/{id}/messages - Add message (used internally)
```

### Phase 2: Frontend Enhancements (~4 days)

#### 2.1 API Client Extensions

Extend `p2a-dioxus/src/api/client.rs`:

```rust
impl ApiClient {
    pub async fn list_conversations(&self) -> Result<Vec<Conversation>, ApiError> {
        self.get("/api/conversations").await
    }

    pub async fn create_conversation(&self, title: &str) -> Result<Conversation, ApiError> {
        self.post("/api/conversations", &CreateConversationRequest { title }).await
    }

    pub async fn get_conversation(&self, id: &str) -> Result<ConversationWithMessages, ApiError> {
        self.get(&format!("/api/conversations/{}", id)).await
    }

    pub async fn update_conversation(&self, id: &str, update: &UpdateConversation) -> Result<Conversation, ApiError> {
        self.put(&format!("/api/conversations/{}", id), update).await
    }

    pub async fn delete_conversation(&self, id: &str) -> Result<(), ApiError> {
        self.delete(&format!("/api/conversations/{}", id)).await
    }
}
```

#### 2.2 State Management Updates

Extend `src/state/chat.rs`:

```rust
#[derive(Clone)]
pub struct ChatState {
    pub current_conversation_id: Option<String>,
    pub conversations: Vec<Conversation>,
    pub messages: Vec<Message>,
    pub is_streaming: bool,
    // ... existing fields
}

impl ChatState {
    pub async fn load_conversation(&mut self, id: &str, client: &ApiClient) {
        match client.get_conversation(id).await {
            Ok(conv) => {
                self.current_conversation_id = Some(conv.conversation.id.clone());
                self.messages = conv.messages;
            }
            Err(e) => {
                tracing::error!("Failed to load conversation: {}", e);
            }
        }
    }

    pub fn switch_conversation(&mut self, id: &str) {
        if self.current_conversation_id.as_deref() != Some(id) {
            self.current_conversation_id = Some(id.to_string());
            self.messages.clear(); // Will be loaded async
        }
    }
}
```

#### 2.3 Auto-Save Current Session

Messages are automatically persisted as they're sent/received:

```rust
// In chat handling code
async fn handle_message_complete(msg: &Message, client: &ApiClient, conv_id: &str) {
    // Save to backend
    if let Err(e) = client.save_message(conv_id, msg).await {
        tracing::error!("Failed to persist message: {}", e);
    }
}
```

### Phase 3: Testing & Polish (~3 days)

#### 3.1 Test Suite

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conversation_crud() {
        let db = DbConnection::connect("memory").await.unwrap();

        // Create
        let conv = db.create_conversation("session-1", "Test Chat").await.unwrap();
        assert_eq!(conv.title, "Test Chat");

        // Read
        let loaded = db.get_conversation(&conv.id).await.unwrap();
        assert_eq!(loaded.id, conv.id);

        // Update
        db.update_conversation(&conv.id, "New Title", false).await.unwrap();
        let updated = db.get_conversation(&conv.id).await.unwrap();
        assert_eq!(updated.title, "New Title");

        // Delete
        db.delete_conversation(&conv.id).await.unwrap();
        assert!(db.get_conversation(&conv.id).await.is_err());
    }

    #[tokio::test]
    async fn test_message_persistence() {
        let db = DbConnection::connect("memory").await.unwrap();
        let conv = db.create_conversation("session-1", "Test").await.unwrap();

        // Add messages
        let msg1 = db.add_message(&conv.id, "user", "Hello").await.unwrap();
        let msg2 = db.add_message(&conv.id, "assistant", "Hi there!").await.unwrap();

        // Load conversation with messages
        let loaded = db.get_conversation_with_messages(&conv.id).await.unwrap();
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0].content, "Hello");
    }
}
```

#### 3.2 Integration Testing

- Test session persistence across server restarts
- Test message ordering and timestamps
- Test concurrent access from multiple browser tabs

---

## Open Questions

### 1. Authentication Strategy

**Question**: Anonymous sessions vs user accounts?

**Options**:
1. **Anonymous (Current)**: Session ID = user identity, no login required
2. **Optional Auth**: Anonymous by default, optional account for cross-device sync

**Recommendation**: Start with anonymous, design schema to support optional accounts later.

### 2. Offline Support

**Question**: Should the web app work offline?

**Considerations**:
- LLM chat requires internet (unless local Ollama)
- Analytics (p2a-core) requires server (in web mode)
- Could cache recent messages in localStorage for viewing

**Recommendation**: Online-only for v1, consider offline viewing cache later.

### 3. Data Retention

**Question**: How long to keep conversation history?

**Options**:
- Unlimited (user manages)
- Auto-archive after 30 days
- Storage quota per session

**Recommendation**: Unlimited for v1, add archive/delete UI.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SurrealDB RocksDB issues | Low | Medium | Well-tested backend, fallback to in-memory |
| Schema migrations | Medium | Medium | Use versioned migrations, test thoroughly |
| Bundle size growth | Low | Low | SurrealDB is server-side only |
| Learning curve | Low | Low | SurrealQL is SQL-like, team knows Rust |

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Backend DB Layer | ~5 days | None |
| Phase 2: Frontend Enhancements | ~4 days | Phase 1 |
| Phase 3: Testing & Polish | ~3 days | Phase 2 |
| **Total** | **~12 days** | |

---

## Success Criteria

1. **Persistence**: Conversations survive server restarts
2. **Current Session**: Messages auto-save without user action
3. **Performance**: Page load < 3s, message send < 100ms
4. **Reliability**: No data loss under normal operation

---

## Appendix A: Full SurrealQL Schema

```surql
-- prompt2analytics SurrealDB Schema
-- Version: 1.0
-- Target: p2a-mcp embedded database (RocksDB backend)

-- ====================
-- Namespace and Database
-- ====================
DEFINE NAMESPACE p2a;
USE NS p2a;
DEFINE DATABASE analytics;
USE DB analytics;

-- ====================
-- Conversations
-- ====================
DEFINE TABLE conversations SCHEMAFULL;
DEFINE FIELD id ON conversations TYPE string;
DEFINE FIELD session_id ON conversations TYPE string;
DEFINE FIELD title ON conversations TYPE string;
DEFINE FIELD created_at ON conversations TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON conversations TYPE datetime DEFAULT time::now();
DEFINE FIELD is_archived ON conversations TYPE bool DEFAULT false;
DEFINE FIELD message_count ON conversations TYPE int DEFAULT 0;
DEFINE FIELD last_message_preview ON conversations TYPE option<string>;

DEFINE INDEX idx_conversations_session ON conversations COLUMNS session_id;
DEFINE INDEX idx_conversations_updated ON conversations COLUMNS updated_at DESC;

-- ====================
-- Messages
-- ====================
DEFINE TABLE messages SCHEMAFULL;
DEFINE FIELD id ON messages TYPE string;
DEFINE FIELD conversation_id ON messages TYPE string;
DEFINE FIELD role ON messages TYPE string ASSERT $value IN ['user', 'assistant', 'system'];
DEFINE FIELD content ON messages TYPE string;
DEFINE FIELD created_at ON messages TYPE datetime DEFAULT time::now();
DEFINE FIELD token_count ON messages TYPE option<int>;
DEFINE FIELD model ON messages TYPE option<string>;
DEFINE FIELD finish_reason ON messages TYPE option<string>;

DEFINE INDEX idx_messages_conversation ON messages COLUMNS conversation_id;
DEFINE INDEX idx_messages_created ON messages COLUMNS conversation_id, created_at;

-- ====================
-- Tool Calls
-- ====================
DEFINE TABLE tool_calls SCHEMAFULL;
DEFINE FIELD id ON tool_calls TYPE string;
DEFINE FIELD message_id ON tool_calls TYPE string;
DEFINE FIELD conversation_id ON tool_calls TYPE string;
DEFINE FIELD tool_name ON tool_calls TYPE string;
DEFINE FIELD arguments ON tool_calls TYPE string;  -- JSON-encoded
DEFINE FIELD result ON tool_calls TYPE option<string>;  -- JSON-encoded
DEFINE FIELD error ON tool_calls TYPE option<string>;
DEFINE FIELD status ON tool_calls TYPE string DEFAULT 'pending'
    ASSERT $value IN ['pending', 'running', 'success', 'error'];
DEFINE FIELD started_at ON tool_calls TYPE datetime DEFAULT time::now();
DEFINE FIELD completed_at ON tool_calls TYPE option<datetime>;
DEFINE FIELD duration_ms ON tool_calls TYPE option<int>;

DEFINE INDEX idx_tool_calls_message ON tool_calls COLUMNS message_id;
DEFINE INDEX idx_tool_calls_conversation ON tool_calls COLUMNS conversation_id;
DEFINE INDEX idx_tool_calls_tool ON tool_calls COLUMNS tool_name;

-- ====================
-- Sessions
-- ====================
DEFINE TABLE sessions SCHEMAFULL;
DEFINE FIELD id ON sessions TYPE string;
DEFINE FIELD user_id ON sessions TYPE option<string>;
DEFINE FIELD created_at ON sessions TYPE datetime DEFAULT time::now();
DEFINE FIELD last_accessed ON sessions TYPE datetime DEFAULT time::now();
DEFINE FIELD global_seed ON sessions TYPE option<int>;
DEFINE FIELD active_conversation_id ON sessions TYPE option<string>;
DEFINE FIELD dataset_names ON sessions TYPE array DEFAULT [];  -- List of loaded dataset names
DEFINE FIELD metadata ON sessions TYPE option<object>;

DEFINE INDEX idx_sessions_user ON sessions COLUMNS user_id;
DEFINE INDEX idx_sessions_accessed ON sessions COLUMNS last_accessed DESC;

-- ====================
-- Settings
-- ====================
DEFINE TABLE settings SCHEMAFULL;
DEFINE FIELD id ON settings TYPE string;
DEFINE FIELD session_id ON settings TYPE string;
DEFINE FIELD provider ON settings TYPE string DEFAULT 'ollama';
DEFINE FIELD model ON settings TYPE string DEFAULT 'llama3.1';
DEFINE FIELD api_key_encrypted ON settings TYPE option<string>;
DEFINE FIELD base_url ON settings TYPE option<string>;
DEFINE FIELD temperature ON settings TYPE float DEFAULT 0.7;
DEFINE FIELD max_tokens ON settings TYPE int DEFAULT 4096;
DEFINE FIELD system_prompt ON settings TYPE option<string>;
DEFINE FIELD updated_at ON settings TYPE datetime DEFAULT time::now();

DEFINE INDEX idx_settings_session ON settings COLUMNS session_id UNIQUE;

-- ====================
-- Dataset Metadata (not the actual data)
-- ====================
DEFINE TABLE dataset_meta SCHEMAFULL;
DEFINE FIELD id ON dataset_meta TYPE string;
DEFINE FIELD session_id ON dataset_meta TYPE string;
DEFINE FIELD name ON dataset_meta TYPE string;
DEFINE FIELD source_path ON dataset_meta TYPE option<string>;
DEFINE FIELD source_type ON dataset_meta TYPE string;  -- csv, parquet, json, etc.
DEFINE FIELD row_count ON dataset_meta TYPE int;
DEFINE FIELD column_count ON dataset_meta TYPE int;
DEFINE FIELD column_names ON dataset_meta TYPE array;
DEFINE FIELD loaded_at ON dataset_meta TYPE datetime DEFAULT time::now();
DEFINE FIELD file_size_bytes ON dataset_meta TYPE option<int>;

DEFINE INDEX idx_dataset_meta_session ON dataset_meta COLUMNS session_id;
DEFINE INDEX idx_dataset_meta_name ON dataset_meta COLUMNS session_id, name UNIQUE;
```

---

## Appendix B: Rust Model Definitions

```rust
// crates/p2a-mcp/src/db/models.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub session_id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_archived: bool,
    pub message_count: i32,
    pub last_message_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub role: MessageRole,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub token_count: Option<i32>,
    pub model: Option<String>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub message_id: String,
    pub conversation_id: String,
    pub tool_name: String,
    pub arguments: String,  // JSON
    pub result: Option<String>,  // JSON
    pub error: Option<String>,
    pub status: ToolCallStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolCallStatus {
    Pending,
    Running,
    Success,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub id: String,
    pub session_id: String,
    pub provider: String,
    pub model: String,
    pub api_key_encrypted: Option<String>,
    pub base_url: Option<String>,
    pub temperature: f64,
    pub max_tokens: i32,
    pub system_prompt: Option<String>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMeta {
    pub id: String,
    pub session_id: String,
    pub name: String,
    pub source_path: Option<String>,
    pub source_type: String,
    pub row_count: i32,
    pub column_count: i32,
    pub column_names: Vec<String>,
    pub loaded_at: DateTime<Utc>,
    pub file_size_bytes: Option<i64>,
}
```

---

## References

- [SurrealDB Rust SDK Documentation](https://surrealdb.com/docs/integration/sdks/rust)
- [SurrealDB Embedded Guide](https://surrealdb.com/docs/surrealdb/introduction/start/embedded)
- [Dioxus Documentation](https://dioxuslabs.com/learn/0.6/)
- [DIOXUS_MIGRATION_FEASIBILITY.md](./DIOXUS_MIGRATION_FEASIBILITY.md) - Previous feasibility study
- [CLAUDE.md](../CLAUDE.md) - Project architecture overview
- [SurrealDB WASM Issues](https://github.com/surrealdb/surrealdb/issues/5712) - Transaction and memory issues
