//! Database layer tests
//!
//! Run with: `cargo test -p p2a-mcp --features full db_tests`

use super::connection::DbConnection;
use super::models::MessageRole;

/// Helper to create an in-memory database for testing
async fn test_db() -> DbConnection {
    DbConnection::connect_memory()
        .await
        .expect("Failed to create in-memory database")
}

// ==================== Conversation Tests ====================

#[tokio::test]
async fn test_create_conversation() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Conversation")
        .await
        .expect("Failed to create conversation");

    assert_eq!(conv.session_id, "session-1");
    assert_eq!(conv.title, "Test Conversation");
    assert!(!conv.is_archived);
    assert_eq!(conv.message_count, 0);
    assert!(conv.last_message_preview.is_none());
}

#[tokio::test]
async fn test_get_conversation() {
    let db = test_db().await;

    let created = db
        .create_conversation("session-1", "My Chat")
        .await
        .expect("Failed to create conversation");

    let fetched = db
        .get_conversation(&created.id_string())
        .await
        .expect("Failed to get conversation");

    assert_eq!(fetched.id_string(), created.id_string());
    assert_eq!(fetched.title, "My Chat");
}

#[tokio::test]
async fn test_get_nonexistent_conversation() {
    let db = test_db().await;

    let result = db.get_conversation("nonexistent-id").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_list_conversations() {
    let db = test_db().await;

    // Create multiple conversations
    db.create_conversation("session-1", "Chat 1")
        .await
        .expect("Failed to create conversation 1");
    db.create_conversation("session-1", "Chat 2")
        .await
        .expect("Failed to create conversation 2");
    db.create_conversation("session-2", "Other Session")
        .await
        .expect("Failed to create conversation 3");

    // List conversations for session-1
    let convs = db
        .list_conversations("session-1")
        .await
        .expect("Failed to list conversations");

    assert_eq!(convs.len(), 2);

    // List conversations for session-2
    let convs = db
        .list_conversations("session-2")
        .await
        .expect("Failed to list conversations");

    assert_eq!(convs.len(), 1);
    assert_eq!(convs[0].title, "Other Session");
}

#[tokio::test]
async fn test_update_conversation_title() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Original Title")
        .await
        .expect("Failed to create conversation");

    let updated = db
        .update_conversation_title(&conv.id_string(), "New Title")
        .await
        .expect("Failed to update title");

    assert_eq!(updated.title, "New Title");

    // Verify the update persisted
    let fetched = db
        .get_conversation(&conv.id_string())
        .await
        .expect("Failed to get conversation");

    assert_eq!(fetched.title, "New Title");
}

#[tokio::test]
async fn test_archive_conversation() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "To Archive")
        .await
        .expect("Failed to create conversation");

    assert!(!conv.is_archived);

    // Archive
    let archived = db
        .set_conversation_archived(&conv.id_string(), true)
        .await
        .expect("Failed to archive");

    assert!(archived.is_archived);

    // Unarchive
    let unarchived = db
        .set_conversation_archived(&conv.id_string(), false)
        .await
        .expect("Failed to unarchive");

    assert!(!unarchived.is_archived);
}

#[tokio::test]
async fn test_delete_conversation() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "To Delete")
        .await
        .expect("Failed to create conversation");

    // Add a message to test cascade delete
    db.add_message(&conv.id_string(), "user", "Hello")
        .await
        .expect("Failed to add message");

    // Delete
    db.delete_conversation(&conv.id_string())
        .await
        .expect("Failed to delete conversation");

    // Verify deleted
    let result = db.get_conversation(&conv.id_string()).await;
    assert!(result.is_err());

    // Verify messages also deleted
    let messages = db
        .get_messages(&conv.id_string())
        .await
        .expect("Failed to get messages");
    assert!(messages.is_empty());
}

// ==================== Message Tests ====================

#[tokio::test]
async fn test_add_message() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    let msg = db
        .add_message(&conv.id_string(), "user", "Hello, world!")
        .await
        .expect("Failed to add message");

    assert_eq!(msg.conversation_id, conv.id_string());
    assert_eq!(msg.role, MessageRole::User);
    assert_eq!(msg.content, "Hello, world!");
}

#[tokio::test]
async fn test_add_multiple_messages() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    db.add_message(&conv.id_string(), "user", "Hello")
        .await
        .expect("Failed to add message 1");
    db.add_message(&conv.id_string(), "assistant", "Hi there!")
        .await
        .expect("Failed to add message 2");
    db.add_message(&conv.id_string(), "user", "How are you?")
        .await
        .expect("Failed to add message 3");

    let messages = db
        .get_messages(&conv.id_string())
        .await
        .expect("Failed to get messages");

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].content, "Hello");
    assert_eq!(messages[1].content, "Hi there!");
    assert_eq!(messages[2].content, "How are you?");
}

#[tokio::test]
async fn test_message_roles() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    let user_msg = db
        .add_message(&conv.id_string(), "user", "User message")
        .await
        .expect("Failed to add user message");
    assert_eq!(user_msg.role, MessageRole::User);

    let assistant_msg = db
        .add_message(&conv.id_string(), "assistant", "Assistant message")
        .await
        .expect("Failed to add assistant message");
    assert_eq!(assistant_msg.role, MessageRole::Assistant);

    let system_msg = db
        .add_message(&conv.id_string(), "system", "System message")
        .await
        .expect("Failed to add system message");
    assert_eq!(system_msg.role, MessageRole::System);
}

#[tokio::test]
async fn test_invalid_message_role() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    let result = db
        .add_message(&conv.id_string(), "invalid_role", "Message")
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_clear_messages() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    // Add messages
    db.add_message(&conv.id_string(), "user", "Message 1")
        .await
        .expect("Failed to add message 1");
    db.add_message(&conv.id_string(), "assistant", "Message 2")
        .await
        .expect("Failed to add message 2");

    // Verify messages exist
    let messages = db
        .get_messages(&conv.id_string())
        .await
        .expect("Failed to get messages");
    assert_eq!(messages.len(), 2);

    // Clear messages
    let deleted_count = db
        .clear_messages(&conv.id_string())
        .await
        .expect("Failed to clear messages");
    assert_eq!(deleted_count, 2);

    // Verify messages cleared
    let messages = db
        .get_messages(&conv.id_string())
        .await
        .expect("Failed to get messages");
    assert!(messages.is_empty());

    // Verify conversation still exists but with reset counters
    let conv = db
        .get_conversation(&conv.id_string())
        .await
        .expect("Failed to get conversation");
    assert_eq!(conv.message_count, 0);
    assert!(conv.last_message_preview.is_none());
}

#[tokio::test]
async fn test_conversation_message_count_updates() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    assert_eq!(conv.message_count, 0);

    // Add a message
    db.add_message(&conv.id_string(), "user", "First message")
        .await
        .expect("Failed to add message");

    let conv = db
        .get_conversation(&conv.id_string())
        .await
        .expect("Failed to get conversation");
    assert_eq!(conv.message_count, 1);
    assert_eq!(conv.last_message_preview, Some("First message".to_string()));

    // Add another message
    db.add_message(&conv.id_string(), "assistant", "Second message")
        .await
        .expect("Failed to add message");

    let conv = db
        .get_conversation(&conv.id_string())
        .await
        .expect("Failed to get conversation");
    assert_eq!(conv.message_count, 2);
    assert_eq!(
        conv.last_message_preview,
        Some("Second message".to_string())
    );
}

#[tokio::test]
async fn test_conversation_with_messages() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    db.add_message(&conv.id_string(), "user", "Hello")
        .await
        .expect("Failed to add message 1");
    db.add_message(&conv.id_string(), "assistant", "Hi!")
        .await
        .expect("Failed to add message 2");

    let full = db
        .get_conversation_with_messages(&conv.id_string())
        .await
        .expect("Failed to get conversation with messages");

    assert_eq!(full.conversation.title, "Test Chat");
    assert_eq!(full.messages.len(), 2);
    assert_eq!(full.messages[0].content, "Hello");
    assert_eq!(full.messages[1].content, "Hi!");
}

// ==================== Session Tests ====================

#[tokio::test]
async fn test_upsert_session() {
    let db = test_db().await;

    let session = super::models::DbSession::new("test-session-1".to_string(), Some("user-123".to_string()));

    let created = db
        .upsert_session(&session)
        .await
        .expect("Failed to upsert session");

    assert_eq!(created.user_id, Some("user-123".to_string()));
}

#[tokio::test]
async fn test_upsert_anonymous_session() {
    let db = test_db().await;

    let session = super::models::DbSession::new("test-session-2".to_string(), None);

    let created = db
        .upsert_session(&session)
        .await
        .expect("Failed to upsert session");

    assert!(created.user_id.is_none());
}

#[tokio::test]
async fn test_get_session() {
    let db = test_db().await;

    let session = super::models::DbSession::new("test-session-3".to_string(), Some("user-456".to_string()));

    let created = db
        .upsert_session(&session)
        .await
        .expect("Failed to upsert session");

    let fetched = db
        .get_session(&created.id_string())
        .await
        .expect("Failed to get session")
        .expect("Session should exist");

    assert_eq!(fetched.id_string(), created.id_string());
    assert_eq!(fetched.user_id, Some("user-456".to_string()));
}

#[tokio::test]
async fn test_touch_session() {
    let db = test_db().await;

    let session = super::models::DbSession::new("test-session-4".to_string(), None);

    let created = db
        .upsert_session(&session)
        .await
        .expect("Failed to upsert session");

    let original_accessed = created.last_accessed.clone();

    // Small delay to ensure time difference
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    db.touch_session(&created.id_string())
        .await
        .expect("Failed to touch session");

    let updated = db
        .get_session(&created.id_string())
        .await
        .expect("Failed to get session")
        .expect("Session should exist");

    // last_accessed should be updated (or at least not earlier)
    assert!(updated.last_accessed >= original_accessed);
}

#[tokio::test]
async fn test_delete_session() {
    let db = test_db().await;

    let session = super::models::DbSession::new("test-session-5".to_string(), None);

    let created = db
        .upsert_session(&session)
        .await
        .expect("Failed to upsert session");

    db.delete_session(&created.id_string())
        .await
        .expect("Failed to delete session");

    let result = db.get_session(&created.id_string()).await;
    assert!(result.expect("Should not error").is_none());
}

// ==================== Integration Tests ====================

#[tokio::test]
async fn test_full_conversation_flow() {
    let db = test_db().await;

    // Create a session
    let session_data = super::models::DbSession::new("integration-test-session".to_string(), Some("test-user".to_string()));
    let session = db
        .upsert_session(&session_data)
        .await
        .expect("Failed to upsert session");

    // Create a conversation
    let conv = db
        .create_conversation(&session.id_string(), "Integration Test Chat")
        .await
        .expect("Failed to create conversation");

    // Simulate a chat exchange
    db.add_message(&conv.id_string(), "user", "What is 2+2?")
        .await
        .expect("Failed to add user message");

    db.add_message(&conv.id_string(), "assistant", "2+2 equals 4.")
        .await
        .expect("Failed to add assistant message");

    db.add_message(&conv.id_string(), "user", "Thanks!")
        .await
        .expect("Failed to add follow-up");

    db.add_message(
        &conv.id_string(),
        "assistant",
        "You're welcome! Let me know if you have more questions.",
    )
    .await
    .expect("Failed to add response");

    // Verify the full conversation
    let full = db
        .get_conversation_with_messages(&conv.id_string())
        .await
        .expect("Failed to get conversation with messages");

    assert_eq!(full.conversation.message_count, 4);
    assert_eq!(full.messages.len(), 4);
    assert_eq!(full.messages[0].role, MessageRole::User);
    assert_eq!(full.messages[1].role, MessageRole::Assistant);
    assert_eq!(full.messages[2].role, MessageRole::User);
    assert_eq!(full.messages[3].role, MessageRole::Assistant);

    // Rename the conversation
    db.update_conversation_title(&conv.id_string(), "Math Questions")
        .await
        .expect("Failed to rename");

    let renamed = db
        .get_conversation(&conv.id_string())
        .await
        .expect("Failed to get renamed conversation");
    assert_eq!(renamed.title, "Math Questions");

    // Archive the conversation
    db.set_conversation_archived(&conv.id_string(), true)
        .await
        .expect("Failed to archive");

    let archived = db
        .get_conversation(&conv.id_string())
        .await
        .expect("Failed to get archived conversation");
    assert!(archived.is_archived);

    // List should still include archived
    let all_convs = db
        .list_conversations(&session.id_string())
        .await
        .expect("Failed to list conversations");
    assert_eq!(all_convs.len(), 1);
}

#[tokio::test]
async fn test_multiple_sessions_isolated() {
    let db = test_db().await;

    // Create two sessions
    let session1_data = super::models::DbSession::new("isolation-session-1".to_string(), Some("user-1".to_string()));
    let session1 = db
        .upsert_session(&session1_data)
        .await
        .expect("Failed to upsert session 1");

    let session2_data = super::models::DbSession::new("isolation-session-2".to_string(), Some("user-2".to_string()));
    let session2 = db
        .upsert_session(&session2_data)
        .await
        .expect("Failed to upsert session 2");

    // Create conversations in each session
    db.create_conversation(&session1.id_string(), "Session 1 Chat")
        .await
        .expect("Failed to create conversation 1");

    db.create_conversation(&session2.id_string(), "Session 2 Chat")
        .await
        .expect("Failed to create conversation 2");

    // Verify isolation
    let convs1 = db
        .list_conversations(&session1.id_string())
        .await
        .expect("Failed to list session 1 conversations");
    assert_eq!(convs1.len(), 1);
    assert_eq!(convs1[0].title, "Session 1 Chat");

    let convs2 = db
        .list_conversations(&session2.id_string())
        .await
        .expect("Failed to list session 2 conversations");
    assert_eq!(convs2.len(), 1);
    assert_eq!(convs2[0].title, "Session 2 Chat");
}

#[tokio::test]
async fn test_long_message_preview_truncation() {
    let db = test_db().await;

    let conv = db
        .create_conversation("session-1", "Test Chat")
        .await
        .expect("Failed to create conversation");

    // Add a very long message
    let long_content = "A".repeat(200);
    db.add_message(&conv.id_string(), "user", &long_content)
        .await
        .expect("Failed to add long message");

    let updated = db
        .get_conversation(&conv.id_string())
        .await
        .expect("Failed to get conversation");

    // Preview should be truncated
    let preview = updated.last_message_preview.expect("Should have preview");
    assert!(preview.len() <= 103); // 100 chars + "..."
    assert!(preview.ends_with("..."));
}
