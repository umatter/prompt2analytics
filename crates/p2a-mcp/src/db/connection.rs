//! SurrealDB connection management
//!
//! Provides connection to embedded SurrealDB with RocksDB or in-memory backend.

use std::path::Path;
use std::sync::Arc;

use surrealdb::engine::local::{Db, Mem, RocksDb};
use surrealdb::Surreal;
use thiserror::Error;

/// Database connection errors
#[derive(Debug, Error)]
pub enum DbError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<surrealdb::Error> for DbError {
    fn from(e: surrealdb::Error) -> Self {
        DbError::Query(e.to_string())
    }
}

/// Database connection wrapper
///
/// Supports both RocksDB (persistent) and in-memory backends.
pub struct DbConnection {
    db: Arc<Surreal<Db>>,
}

impl DbConnection {
    /// Connect to a RocksDB database at the given path
    pub async fn connect_rocksdb(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let path = path.as_ref();
        tracing::info!("Connecting to SurrealDB RocksDB at {:?}", path);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                DbError::Connection(format!("Failed to create database directory: {}", e))
            })?;
        }

        let db = Surreal::new::<RocksDb>(path)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        Self::setup(db).await
    }

    /// Connect to an in-memory database (for testing)
    pub async fn connect_memory() -> Result<Self, DbError> {
        tracing::info!("Connecting to SurrealDB in-memory");

        let db = Surreal::new::<Mem>(())
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        Self::setup(db).await
    }

    /// Common setup for both connection types
    async fn setup(db: Surreal<Db>) -> Result<Self, DbError> {
        // Use namespace and database
        db.use_ns("p2a").use_db("analytics").await?;

        // Run schema setup
        Self::setup_schema(&db).await?;

        tracing::info!("SurrealDB connected and schema initialized");

        Ok(Self { db: Arc::new(db) })
    }

    /// Initialize database schema
    async fn setup_schema(db: &Surreal<Db>) -> Result<(), DbError> {
        // Define schema using SurrealQL
        // Using DEFINE TABLE IF NOT EXISTS pattern for idempotency
        let schema = r#"
            -- Conversations table
            DEFINE TABLE IF NOT EXISTS conversations SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON conversations TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON conversations TYPE string;
            DEFINE FIELD IF NOT EXISTS title ON conversations TYPE string;
            DEFINE FIELD IF NOT EXISTS created_at ON conversations TYPE datetime;
            DEFINE FIELD IF NOT EXISTS updated_at ON conversations TYPE datetime;
            DEFINE FIELD IF NOT EXISTS is_archived ON conversations TYPE bool DEFAULT false;
            DEFINE FIELD IF NOT EXISTS message_count ON conversations TYPE int DEFAULT 0;
            DEFINE FIELD IF NOT EXISTS last_message_preview ON conversations TYPE option<string>;
            DEFINE INDEX IF NOT EXISTS idx_conversations_session ON conversations COLUMNS session_id;
            DEFINE INDEX IF NOT EXISTS idx_conversations_id ON conversations COLUMNS id UNIQUE;

            -- Messages table
            DEFINE TABLE IF NOT EXISTS messages SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS conversation_id ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS role ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS content ON messages TYPE string;
            DEFINE FIELD IF NOT EXISTS created_at ON messages TYPE datetime;
            DEFINE FIELD IF NOT EXISTS token_count ON messages TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS model ON messages TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS finish_reason ON messages TYPE option<string>;
            DEFINE INDEX IF NOT EXISTS idx_messages_conversation ON messages COLUMNS conversation_id;
            DEFINE INDEX IF NOT EXISTS idx_messages_id ON messages COLUMNS id UNIQUE;

            -- Tool calls table
            DEFINE TABLE IF NOT EXISTS tool_calls SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS message_id ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS conversation_id ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS tool_name ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS arguments ON tool_calls TYPE string;
            DEFINE FIELD IF NOT EXISTS result ON tool_calls TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS error ON tool_calls TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS status ON tool_calls TYPE string DEFAULT 'pending';
            DEFINE FIELD IF NOT EXISTS started_at ON tool_calls TYPE datetime;
            DEFINE FIELD IF NOT EXISTS completed_at ON tool_calls TYPE option<datetime>;
            DEFINE FIELD IF NOT EXISTS duration_ms ON tool_calls TYPE option<int>;
            DEFINE INDEX IF NOT EXISTS idx_tool_calls_message ON tool_calls COLUMNS message_id;
            DEFINE INDEX IF NOT EXISTS idx_tool_calls_id ON tool_calls COLUMNS id UNIQUE;

            -- Sessions table
            DEFINE TABLE IF NOT EXISTS sessions SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON sessions TYPE string;
            DEFINE FIELD IF NOT EXISTS user_id ON sessions TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS created_at ON sessions TYPE datetime;
            DEFINE FIELD IF NOT EXISTS last_accessed ON sessions TYPE datetime;
            DEFINE FIELD IF NOT EXISTS global_seed ON sessions TYPE option<int>;
            DEFINE FIELD IF NOT EXISTS active_conversation_id ON sessions TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS dataset_names ON sessions TYPE array DEFAULT [];
            DEFINE INDEX IF NOT EXISTS idx_sessions_id ON sessions COLUMNS id UNIQUE;
            DEFINE INDEX IF NOT EXISTS idx_sessions_user ON sessions COLUMNS user_id;

            -- Settings table
            DEFINE TABLE IF NOT EXISTS settings SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON settings TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON settings TYPE string;
            DEFINE FIELD IF NOT EXISTS provider ON settings TYPE string DEFAULT 'ollama';
            DEFINE FIELD IF NOT EXISTS model ON settings TYPE string DEFAULT 'llama3.1';
            DEFINE FIELD IF NOT EXISTS api_key_encrypted ON settings TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS base_url ON settings TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS temperature ON settings TYPE float DEFAULT 0.7;
            DEFINE FIELD IF NOT EXISTS max_tokens ON settings TYPE int DEFAULT 4096;
            DEFINE FIELD IF NOT EXISTS system_prompt ON settings TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS updated_at ON settings TYPE datetime;
            DEFINE INDEX IF NOT EXISTS idx_settings_session ON settings COLUMNS session_id UNIQUE;

            -- Dataset metadata table
            DEFINE TABLE IF NOT EXISTS dataset_meta SCHEMAFULL;
            DEFINE FIELD IF NOT EXISTS id ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS session_id ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS name ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS source_path ON dataset_meta TYPE option<string>;
            DEFINE FIELD IF NOT EXISTS source_type ON dataset_meta TYPE string;
            DEFINE FIELD IF NOT EXISTS row_count ON dataset_meta TYPE int;
            DEFINE FIELD IF NOT EXISTS column_count ON dataset_meta TYPE int;
            DEFINE FIELD IF NOT EXISTS column_names ON dataset_meta TYPE array;
            DEFINE FIELD IF NOT EXISTS loaded_at ON dataset_meta TYPE datetime;
            DEFINE FIELD IF NOT EXISTS file_size_bytes ON dataset_meta TYPE option<int>;
            DEFINE INDEX IF NOT EXISTS idx_dataset_meta_session ON dataset_meta COLUMNS session_id;
            DEFINE INDEX IF NOT EXISTS idx_dataset_meta_id ON dataset_meta COLUMNS id UNIQUE;
        "#;

        db.query(schema).await?;
        Ok(())
    }

    /// Get a reference to the database for queries
    pub fn db(&self) -> &Surreal<Db> {
        &self.db
    }

    /// Check if the database is healthy
    pub async fn health_check(&self) -> Result<bool, DbError> {
        // Try a simple query
        self.db.query("SELECT * FROM sessions LIMIT 1").await?;
        Ok(true)
    }

    /// Get database statistics
    pub async fn stats(&self) -> Result<DbStats, DbError> {
        let mut result = self
            .db
            .query(
                r#"
                RETURN {
                    sessions: (SELECT count() FROM sessions GROUP ALL)[0].count ?? 0,
                    conversations: (SELECT count() FROM conversations GROUP ALL)[0].count ?? 0,
                    messages: (SELECT count() FROM messages GROUP ALL)[0].count ?? 0
                }
            "#,
            )
            .await?;

        #[derive(serde::Deserialize, Default)]
        struct StatsResult {
            sessions: Option<i64>,
            conversations: Option<i64>,
            messages: Option<i64>,
        }

        let stats: Option<StatsResult> = result.take(0)?;
        let stats = stats.unwrap_or_default();

        Ok(DbStats {
            session_count: stats.sessions.unwrap_or(0) as u32,
            conversation_count: stats.conversations.unwrap_or(0) as u32,
            message_count: stats.messages.unwrap_or(0) as u32,
        })
    }
}

impl Clone for DbConnection {
    fn clone(&self) -> Self {
        Self {
            db: Arc::clone(&self.db),
        }
    }
}

/// Database statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct DbStats {
    pub session_count: u32,
    pub conversation_count: u32,
    pub message_count: u32,
}
