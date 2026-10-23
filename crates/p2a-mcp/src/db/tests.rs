//! Database layer tests
//!
//! Run with: `cargo test -p p2a-mcp --features full db_tests`

use super::connection::{DbConnection, DbError, PersistenceLimits, truncate_to};
use super::models::{Message, MessageRole, ToolCall};

/// Helper to create an in-memory database for testing
async fn test_db() -> DbConnection {
    DbConnection::connect_memory()
        .await
        .expect("Failed to create in-memory database")
}

/// An in-memory database with `limits` applied.
async fn limited_db(limits: PersistenceLimits) -> DbConnection {
    test_db().await.with_limits(limits)
}

// ==================== Limits ====================

#[tokio::test]
async fn add_message_refuses_unknown_conversation() {
    let db = test_db().await;
    assert!(matches!(
        db.add_message("no-such-conversation", "user", "x").await,
        Err(DbError::NotFound(_))
    ));
    let message = Message::user("no-such-conversation".into(), "x".into());
    assert!(matches!(
        db.save_message(&message).await,
        Err(DbError::NotFound(_))
    ));
    assert_eq!(db.stats().await.unwrap().message_count, 0);
}

#[tokio::test]
async fn add_message_enforces_size_and_count() {
    let db = limited_db(PersistenceLimits {
        message_bytes: Some(8),
        messages_per_conversation: Some(1),
        ..Default::default()
    })
    .await;
    let conv = db.create_conversation("s1", "t").await.unwrap();
    let id = conv.id_string();
    match db.add_message(&id, "user", "123456789").await {
        Err(DbError::TooLarge(text)) => {
            assert!(
                text.contains("9 bytes") && text.contains("at most 8"),
                "{text}"
            )
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
    db.add_message(&id, "user", "12345678").await.unwrap();
    assert!(matches!(
        db.add_message(&id, "user", "x").await,
        Err(DbError::QuotaExceeded(_))
    ));
    assert!(matches!(
        db.save_message(&Message::user(id.clone(), "x".into()))
            .await,
        Err(DbError::QuotaExceeded(_))
    ));
    assert_eq!(db.get_messages(&id).await.unwrap().len(), 1);
}

#[tokio::test]
async fn create_conversation_enforces_per_session_count() {
    let db = limited_db(PersistenceLimits {
        message_bytes: Some(8),
        conversations_per_session: Some(2),
        ..Default::default()
    })
    .await;
    assert!(matches!(
        db.create_conversation("s1", "123456789").await,
        Err(DbError::TooLarge(_))
    ));
    db.create_conversation("s1", "a").await.unwrap();
    db.create_conversation("s1", "b").await.unwrap();
    assert!(matches!(
        db.create_conversation("s1", "c").await,
        Err(DbError::QuotaExceeded(_))
    ));
    // The count is per session.
    db.create_conversation("s2", "a").await.unwrap();
    assert_eq!(db.count_conversations("s1").await.unwrap(), 2);
    // A new title counts against message_bytes too.
    let conv = db.list_conversations("s1").await.unwrap().remove(0);
    assert!(matches!(
        db.update_conversation_title(&conv.id_string(), "123456789")
            .await,
        Err(DbError::TooLarge(_))
    ));
}

#[test]
fn truncate_to_cuts_at_char_boundaries() {
    assert_eq!(truncate_to("short", 20), "short");
    // 21 - 12 (suffix) = 9 bytes, inside the fifth "ä": cut to 8.
    let cut = truncate_to(&"ä".repeat(20), 21);
    assert_eq!(cut, format!("{}\n[truncated]", "ä".repeat(4)));
    // No room for the suffix: nothing is kept.
    assert_eq!(truncate_to(&"x".repeat(50), 11), "");
    assert_eq!(truncate_to(&"x".repeat(50), 12), "\n[truncated]");
}

#[tokio::test]
async fn update_message_content_truncates_at_char_boundary() {
    let db = limited_db(PersistenceLimits {
        message_bytes: Some(20),
        ..Default::default()
    })
    .await;
    let conv = db.create_conversation("s1", "t").await.unwrap();
    let msg = db
        .add_message(&conv.id_string(), "assistant", "")
        .await
        .unwrap();
    let stored = db
        .update_message_content(&msg.id_string(), &"ä".repeat(20))
        .await
        .unwrap();
    assert!(stored.content.len() <= 20, "{}", stored.content.len());
    assert!(
        stored.content.ends_with("[truncated]"),
        "{}",
        stored.content
    );
    assert!(stored.content.starts_with('ä'));
    assert_eq!(
        db.get_message(&msg.id_string()).await.unwrap().content,
        stored.content
    );
}

#[tokio::test]
async fn tool_call_text_is_truncated() {
    let db = limited_db(PersistenceLimits {
        message_bytes: Some(64),
        ..Default::default()
    })
    .await;
    let conv = db.create_conversation("s1", "t").await.unwrap();
    let msg = db
        .add_message(&conv.id_string(), "assistant", "")
        .await
        .unwrap();
    let new_call = |arguments: String| {
        ToolCall::new(
            msg.id_string(),
            conv.id_string(),
            "regression_ols".into(),
            arguments,
        )
    };
    let tc = db
        .create_tool_call(&new_call("é".repeat(100)))
        .await
        .unwrap();
    assert!(tc.arguments.len() <= 64 && tc.arguments.ends_with("[truncated]"));
    let done = db
        .complete_tool_call(&tc.id_string(), &"r".repeat(500), 3)
        .await
        .unwrap();
    let result = done.result.unwrap();
    assert!(result.len() <= 64 && result.ends_with("[truncated]"));
    let tc2 = db.create_tool_call(&new_call("{}".into())).await.unwrap();
    assert_eq!(tc2.arguments, "{}");
    let failed = db
        .fail_tool_call(&tc2.id_string(), &"e".repeat(500), 3)
        .await
        .unwrap();
    assert!(failed.error.unwrap().len() <= 64);
    // The conversation must exist.
    let ghost = ToolCall::new(msg.id_string(), "ghost".into(), "x".into(), "{}".into());
    assert!(matches!(
        db.create_tool_call(&ghost).await,
        Err(DbError::NotFound(_))
    ));
}

#[tokio::test]
async fn memory_admission_refuses_writes() {
    if p2a_core::get_process_memory().is_none() {
        return; // the platform does not report the process memory
    }
    let open = test_db().await;
    let conv = open.create_conversation("s1", "t").await.unwrap();
    let msg = open
        .add_message(&conv.id_string(), "assistant", "")
        .await
        .unwrap();
    let db = open.clone().with_limits(PersistenceLimits {
        rss_soft_limit_bytes: Some(1),
        ..Default::default()
    });
    assert!(matches!(
        db.create_conversation("s1", "t").await,
        Err(DbError::MemoryPressure)
    ));
    assert!(matches!(
        db.add_message(&conv.id_string(), "user", "x").await,
        Err(DbError::MemoryPressure)
    ));
    assert!(matches!(
        db.save_message(&Message::user(conv.id_string(), "x".into()))
            .await,
        Err(DbError::MemoryPressure)
    ));
    let call = ToolCall::new(msg.id_string(), conv.id_string(), "x".into(), "{}".into());
    assert!(matches!(
        db.create_tool_call(&call).await,
        Err(DbError::MemoryPressure)
    ));
    // The server's final answer to a turn that already ran still lands.
    db.update_message_content(&msg.id_string(), "answer")
        .await
        .unwrap();
    assert_eq!(open.stats().await.unwrap().message_count, 1);
}

#[tokio::test]
async fn session_db_bytes_bounds_the_session_total() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100),
        ..Default::default()
    })
    .await;
    let c = db.create_conversation("s1", "t").await.unwrap(); // 1
    db.add_message(&c.id_string(), "user", &"a".repeat(60))
        .await
        .unwrap(); // 61
    match db
        .add_message(&c.id_string(), "user", &"b".repeat(40))
        .await
    {
        Err(DbError::QuotaExceeded(text)) => {
            assert!(
                text.contains("61 bytes") && text.contains("100 bytes"),
                "{text}"
            )
        } // 101 > 100
        other => panic!("expected QuotaExceeded, got {other:?}"),
    }
    db.add_message(&c.id_string(), "user", &"c".repeat(39))
        .await
        .unwrap(); // exactly 100
    assert_eq!(db.session_stored_bytes("s1").await.unwrap(), 100);
    let other = db.create_conversation("s2", "t").await.unwrap(); // own budget
    db.add_message(&other.id_string(), "user", &"d".repeat(60))
        .await
        .unwrap();

    // Server-produced text is truncated to the room left, never refused.
    let c3 = db.create_conversation("s3", "t").await.unwrap(); // 1
    let m = db
        .add_message(&c3.id_string(), "assistant", "")
        .await
        .unwrap();
    let stored = db
        .update_message_content(&m.id_string(), &"y".repeat(200))
        .await
        .unwrap();
    assert!(
        stored.content.len() <= 99 && stored.content.ends_with("[truncated]"),
        "{}",
        stored.content.len()
    );
    assert!(matches!(
        db.add_message(&c3.id_string(), "user", "z").await,
        Err(DbError::QuotaExceeded(_))
    ));
    db.delete_conversation(&c3.id_string()).await.unwrap(); // frees s3's bytes
    let c4 = db.create_conversation("s3", "t").await.unwrap();
    db.add_message(&c4.id_string(), "user", &"e".repeat(90))
        .await
        .unwrap();

    // Tool-call text counts and is truncated to the room left.
    let c5 = db.create_conversation("s5", "t").await.unwrap(); // 1
    let m5 = db
        .add_message(&c5.id_string(), "assistant", "")
        .await
        .unwrap();
    let tc = db
        .create_tool_call(&ToolCall::new(
            m5.id_string(),
            c5.id_string(),
            "regression_ols".into(),
            "x".repeat(50),
        ))
        .await
        .unwrap(); // 51
    let done = db
        .complete_tool_call(&tc.id_string(), &"r".repeat(500), 3)
        .await
        .unwrap();
    assert!(done.result.as_deref().unwrap().len() <= 49);
    assert_eq!(db.session_stored_bytes("s5").await.unwrap(), 100);

    // Clearing the conversation releases its messages' and tool calls'
    // bytes from the counter and keeps its title's.
    db.clear_messages(&c5.id_string()).await.unwrap();
    assert_eq!(db.session_stored_bytes("s5").await.unwrap(), 1);
    assert_eq!(db.session_bytes_entry("s5"), Some(1));
    db.add_message(&c5.id_string(), "user", &"f".repeat(99))
        .await
        .unwrap();

    // Limits off: no accounting at all.
    let off = test_db().await;
    let c = off.create_conversation("s1", "t").await.unwrap();
    off.add_message(&c.id_string(), "user", &"f".repeat(10_000))
        .await
        .unwrap();
}

/// Deleting or clearing a conversation subtracts its bytes from the
/// session's counter instead of dropping the counter, so the next write does
/// not re-read the whole session (which costs one row read per stored
/// message).
#[tokio::test]
async fn deleting_a_conversation_releases_its_bytes_without_a_reseed() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(1000),
        ..Default::default()
    })
    .await;
    let keep = db.create_conversation("s1", "keep").await.unwrap(); // 4
    db.add_message(&keep.id_string(), "user", &"a".repeat(100))
        .await
        .unwrap(); // 104
    let gone = db.create_conversation("s1", "gone").await.unwrap(); // 108
    let m = db
        .add_message(&gone.id_string(), "assistant", "")
        .await
        .unwrap();
    let call = ToolCall::new(
        m.id_string(),
        gone.id_string(),
        "regression_ols".into(),
        "y".repeat(50),
    );
    db.create_tool_call(&call).await.unwrap(); // 158
    db.add_message(&gone.id_string(), "user", &"b".repeat(200))
        .await
        .unwrap(); // 358
    assert_eq!(db.session_bytes_entry("s1"), Some(358));

    db.delete_conversation(&gone.id_string()).await.unwrap();
    assert_eq!(db.session_bytes_entry("s1"), Some(104));

    let c = db.create_conversation("s1", "c").await.unwrap(); // 105
    db.add_message(&c.id_string(), "user", &"c".repeat(10))
        .await
        .unwrap(); // 115
    db.clear_messages(&c.id_string()).await.unwrap();
    assert_eq!(db.session_bytes_entry("s1"), Some(105));
    assert_eq!(db.session_stored_bytes("s1").await.unwrap(), 105);

    // Deleting the session's rows drops the counter.
    db.delete_session("s1").await.unwrap();
    assert_eq!(db.session_bytes_entry("s1"), None);
}

/// A conversation of `session_id` holding twenty messages and a tool call;
/// returns its id.
async fn filled_conversation(db: &DbConnection, session_id: &str, title: &str) -> String {
    let conv = db.create_conversation(session_id, title).await.unwrap();
    let id = conv.id_string();
    for i in 0..20 {
        db.add_message(&id, "user", &format!("message {i}: {}", "x".repeat(40 + i)))
            .await
            .unwrap();
    }
    let m = db.add_message(&id, "assistant", "").await.unwrap();
    let call = ToolCall::new(
        m.id_string(),
        id.clone(),
        "regression_ols".into(),
        "a".repeat(40),
    );
    db.create_tool_call(&call).await.unwrap();
    id
}

/// The counter of `session_id` equals what the store holds.
async fn assert_counter_matches_store(db: &DbConnection, session_id: &str) {
    let stored = db.session_stored_bytes(session_id).await.unwrap();
    assert_eq!(db.session_bytes_entry(session_id), Some(stored));
}

/// Two concurrent deletes (or clears) of one conversation release its bytes
/// once: each call releases only what its own statements removed, so the
/// counter never drops below what the store holds.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_deletes_release_a_conversations_bytes_once() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100_000),
        ..Default::default()
    })
    .await;
    let keep = db.create_conversation("s1", "keep").await.unwrap();
    db.add_message(&keep.id_string(), "user", &"k".repeat(100))
        .await
        .unwrap();

    let gone = filled_conversation(&db, "s1", "gone").await;
    assert_counter_matches_store(&db, "s1").await; // seeded
    let (a, b) = tokio::join!(db.delete_conversation(&gone), db.delete_conversation(&gone));
    // One deletes it; the other, in its turn after the first, finds nothing.
    assert!(
        matches!(
            (&a, &b),
            (Ok(()), Err(DbError::NotFound(_))) | (Err(DbError::NotFound(_)), Ok(()))
        ),
        "{a:?} / {b:?}"
    );
    assert_counter_matches_store(&db, "s1").await;

    let cleared = filled_conversation(&db, "s1", "cleared").await;
    assert_counter_matches_store(&db, "s1").await;
    let (a, b) = tokio::join!(db.clear_messages(&cleared), db.clear_messages(&cleared));
    assert!(a.is_ok() && b.is_ok(), "{a:?} / {b:?}");
    assert_counter_matches_store(&db, "s1").await;
    assert!(db.get_messages(&cleared).await.unwrap().is_empty());
    assert_eq!(db.store_turns_in_use(), 0);
}

/// Without the byte budget, on RocksDB (which fails one of two conflicting
/// transactions): two parallel deletes of one conversation take turns, so
/// the second finds nothing instead of failing on a write conflict.
#[tokio::test(flavor = "multi_thread")]
async fn parallel_deletes_on_rocksdb_do_not_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let db = DbConnection::connect_rocksdb(dir.path().join("db"))
        .await
        .unwrap();
    for _ in 0..5 {
        let gone = filled_conversation(&db, "s1", "gone").await;
        let (a, b) = tokio::join!(db.delete_conversation(&gone), db.delete_conversation(&gone));
        assert!(
            matches!(
                (&a, &b),
                (Ok(()), Err(DbError::NotFound(_))) | (Err(DbError::NotFound(_)), Ok(()))
            ),
            "{a:?} / {b:?}"
        );
        let cleared = filled_conversation(&db, "s1", "cleared").await;
        let (a, b) = tokio::join!(db.clear_messages(&cleared), db.clear_messages(&cleared));
        assert!(a.is_ok() && b.is_ok(), "{a:?} / {b:?}");
    }
    assert_eq!(db.store_turns_in_use(), 0);
}

/// A store turn leaves the map with its last holder or waiter, also when
/// the waiting call is cancelled.
#[tokio::test]
async fn store_turns_leave_the_map() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100),
        ..Default::default()
    })
    .await;
    let first = db.conversation_turn("c").await;
    {
        let waiting = db.conversation_turn("c");
        tokio::pin!(waiting);
        // It cannot finish while `first` is held; the wait is then cancelled.
        let polled = tokio::time::timeout(std::time::Duration::from_millis(20), &mut waiting).await;
        assert!(polled.is_err());
    }
    assert_eq!(db.store_turns_in_use(), 1);
    drop(first);
    assert_eq!(db.store_turns_in_use(), 0);
    let again = db.conversation_turn("c").await;
    assert_eq!(db.store_turns_in_use(), 1);
    drop(again);
    assert_eq!(db.store_turns_in_use(), 0);
}

/// Concurrent posts to a conversation one message below its cap: the count
/// check and the write take turns, so exactly one of them gets in.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_adds_stop_at_the_message_cap() {
    let db = limited_db(PersistenceLimits {
        messages_per_conversation: Some(3),
        ..Default::default()
    })
    .await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();
    db.add_message(&id, "user", "first").await.unwrap();
    db.add_message(&id, "assistant", "second").await.unwrap();
    let saved = Message::user(id.clone(), "saved".into());
    let (a, b, c, d) = tokio::join!(
        db.add_message(&id, "user", "a"),
        db.add_message(&id, "user", "b"),
        db.add_message(&id, "user", "c"),
        db.save_message(&saved),
    );
    let results = [a.map(|_| ()), b.map(|_| ()), c.map(|_| ()), d.map(|_| ())];
    assert_eq!(
        results.iter().filter(|r| r.is_ok()).count(),
        1,
        "{results:?}"
    );
    assert!(
        results
            .iter()
            .all(|r| r.is_ok() || matches!(r, Err(DbError::QuotaExceeded(_)))),
        "{results:?}"
    );
    assert_eq!(db.get_messages(&id).await.unwrap().len(), 3);
    assert_eq!(db.store_turns_in_use(), 0);
}

/// Concurrent creations in a session one conversation below its cap: exactly
/// one gets in.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_creates_stop_at_the_conversation_cap() {
    let db = limited_db(PersistenceLimits {
        conversations_per_session: Some(2),
        ..Default::default()
    })
    .await;
    db.create_conversation("s1", "first").await.unwrap();
    let (a, b, c) = tokio::join!(
        db.create_conversation("s1", "a"),
        db.create_conversation("s1", "b"),
        db.create_conversation("s1", "c"),
    );
    let results = [a.map(|_| ()), b.map(|_| ()), c.map(|_| ())];
    assert_eq!(
        results.iter().filter(|r| r.is_ok()).count(),
        1,
        "{results:?}"
    );
    assert!(
        results
            .iter()
            .all(|r| r.is_ok() || matches!(r, Err(DbError::QuotaExceeded(_)))),
        "{results:?}"
    );
    assert_eq!(db.count_conversations("s1").await.unwrap(), 2);
    assert_eq!(db.store_turns_in_use(), 0);
}

/// Without the byte budget, a failed row delete is reported: the
/// conversation is not deleted with its messages left behind, and a clear
/// does not claim success.
#[tokio::test]
async fn a_failed_row_delete_is_reported() {
    let db = test_db().await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();
    db.add_message(&id, "user", "kept").await.unwrap();
    // Every delete of a message now fails inside the store.
    db.db()
        .query(
            "DEFINE EVENT refuse_delete ON TABLE messages WHEN $event = 'DELETE' \
             THEN { THROW 'refused' }",
        )
        .await
        .unwrap()
        .check()
        .unwrap();

    assert!(db.delete_conversation(&id).await.is_err());
    assert!(db.get_conversation(&id).await.is_ok());
    assert!(db.clear_messages(&id).await.is_err());
    assert_eq!(db.get_messages(&id).await.unwrap().len(), 1);
}

/// A failed delete of a session's dataset metadata is reported, and the
/// session row stays: DELETE /api/sessions/{id} does not answer 204 while
/// the metadata (with its source path) is left behind.
#[tokio::test]
async fn a_failed_dataset_meta_delete_keeps_the_session() {
    let db = test_db().await;
    db.set_session_last_accessed("s1", chrono::Utc::now())
        .await
        .unwrap();
    let meta = super::models::DatasetMeta::new(
        "s1".into(),
        "d".into(),
        "csv".into(),
        3,
        1,
        vec!["x".into()],
    );
    db.save_dataset_meta(&meta).await.unwrap();
    db.db()
        .query(
            "DEFINE EVENT refuse_delete ON TABLE dataset_meta WHEN $event = 'DELETE' \
             THEN { THROW 'refused' }",
        )
        .await
        .unwrap()
        .check()
        .unwrap();

    assert!(db.delete_session("s1").await.is_err());
    assert!(db.get_session("s1").await.unwrap().is_some());
    assert_eq!(db.get_datasets_for_session("s1").await.unwrap().len(), 1);
}

/// The message cap counts the conversation's message rows, not its stored
/// `message_count`: a counter left one or more below the truth (a recount
/// that lost a write) does not let a message past the cap.
#[tokio::test]
async fn the_message_cap_counts_the_stored_messages() {
    let db = limited_db(PersistenceLimits {
        messages_per_conversation: Some(2),
        ..Default::default()
    })
    .await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();
    db.add_message(&id, "user", "first").await.unwrap();
    db.add_message(&id, "assistant", "second").await.unwrap();
    db.db()
        .query("UPDATE conversations SET message_count = 0 WHERE id = $id")
        .bind((
            "id",
            surrealdb::RecordId::from(("conversations", id.as_str())),
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
    assert_eq!(db.get_conversation(&id).await.unwrap().message_count, 0);

    assert!(matches!(
        db.add_message(&id, "user", "third").await,
        Err(DbError::QuotaExceeded(_))
    ));
    let saved = Message::user(id.clone(), "saved".into());
    assert!(matches!(
        db.save_message(&saved).await,
        Err(DbError::QuotaExceeded(_))
    ));
    assert_eq!(db.count_messages(&id).await.unwrap(), 2);
}

/// A failed write of a conversation's counters is reported: a message add,
/// a content update and a clear do not answer Ok when the conversation row
/// cannot be updated. The error comes after the message row was written, so
/// the failed add's message is stored.
#[tokio::test]
async fn a_failed_counter_write_is_reported() {
    let db = test_db().await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();
    let kept = db.add_message(&id, "user", "kept").await.unwrap();
    // Every update of a conversation row now fails inside the store.
    db.db()
        .query(
            "DEFINE EVENT refuse_update ON TABLE conversations WHEN $event = 'UPDATE' \
             THEN { THROW 'refused' }",
        )
        .await
        .unwrap()
        .check()
        .unwrap();

    assert!(db.add_message(&id, "user", "more").await.is_err());
    assert_eq!(db.count_messages(&id).await.unwrap(), 2);
    assert!(
        db.update_message_content(&kept.id_string(), "changed")
            .await
            .is_err()
    );
    assert!(db.clear_messages(&id).await.is_err());
    assert_eq!(db.store_turns_in_use(), 0);
}

/// Whether `n` calls hold or wait for store turn `key`, polled a bounded
/// number of times (no timing assertion: a slow machine only polls longer).
async fn turn_has_users(db: &DbConnection, key: &str, n: usize) -> bool {
    for _ in 0..500 {
        if db.store_turn_users(key) == n {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    false
}

/// A content update runs in the conversation's turn: while another call
/// holds the turn, the update waits for it with neither the message row nor
/// the conversation's counters changed, and both change once it has run.
#[tokio::test(flavor = "multi_thread")]
async fn a_content_update_recounts_in_the_conversations_turn() {
    let db = test_db().await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();
    let message_id = db
        .add_message(&id, "user", "kept")
        .await
        .unwrap()
        .id_string();

    let turn = db.conversation_turn(&id).await;
    let update = tokio::spawn({
        let (db, message_id) = (db.clone(), message_id.clone());
        async move { db.update_message_content(&message_id, "changed").await }
    });
    // The update is waiting for the turn held here.
    assert!(turn_has_users(&db, &id, 2).await);
    assert_eq!(db.get_message(&message_id).await.unwrap().content, "kept");
    let conv = db.get_conversation(&id).await.unwrap();
    assert_eq!(conv.last_message_preview.as_deref(), Some("kept"));

    drop(turn);
    update.await.unwrap().unwrap();
    assert_eq!(
        db.get_message(&message_id).await.unwrap().content,
        "changed"
    );
    let conv = db.get_conversation(&id).await.unwrap();
    assert_eq!(conv.last_message_preview.as_deref(), Some("changed"));
    assert_eq!(conv.message_count, 1);
}

/// With the byte budget, a content update whose message went before its
/// turn (a clear that ran between the update's read and its turn) is
/// `NotFound` and counts no bytes for the write that found no row.
#[tokio::test(flavor = "multi_thread")]
async fn a_content_update_after_a_clear_counts_no_bytes() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(1000),
        ..Default::default()
    })
    .await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();
    // The server's empty placeholder for its answer.
    let message_id = db
        .add_message(&id, "assistant", "")
        .await
        .unwrap()
        .id_string();
    assert_counter_matches_store(&db, "s1").await; // seeded

    let turn = db.conversation_turn(&id).await;
    let update = tokio::spawn({
        let (db, message_id) = (db.clone(), message_id.clone());
        async move {
            db.update_message_content(&message_id, "a final answer")
                .await
        }
    });
    assert!(turn_has_users(&db, &id, 2).await);
    // The clear's delete, in the turn the update waits for (the placeholder
    // held no bytes, so there is nothing to release).
    let _: Option<Message> = db
        .db()
        .delete(("messages", message_id.as_str()))
        .await
        .unwrap();
    drop(turn);

    assert!(matches!(update.await.unwrap(), Err(DbError::NotFound(_))));
    assert_counter_matches_store(&db, "s1").await;
}

/// A title change and an archive take the conversation's turn, like every
/// write of its row: while another call holds it, both wait with the row
/// unchanged.
#[tokio::test(flavor = "multi_thread")]
async fn title_and_archive_writes_take_the_conversations_turn() {
    let db = test_db().await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();

    let turn = db.conversation_turn(&id).await;
    let title = tokio::spawn({
        let (db, id) = (db.clone(), id.clone());
        async move { db.update_conversation_title(&id, "renamed").await }
    });
    assert!(turn_has_users(&db, &id, 2).await);
    let archive = tokio::spawn({
        let (db, id) = (db.clone(), id.clone());
        async move { db.set_conversation_archived(&id, true).await }
    });
    assert!(turn_has_users(&db, &id, 3).await);
    let conv = db.get_conversation(&id).await.unwrap();
    assert_eq!((conv.title.as_str(), conv.is_archived), ("t", false));

    drop(turn);
    title.await.unwrap().unwrap();
    archive.await.unwrap().unwrap();
    let conv = db.get_conversation(&id).await.unwrap();
    assert_eq!((conv.title.as_str(), conv.is_archived), ("renamed", true));
    assert_eq!(db.store_turns_in_use(), 0);
}

/// Store a message for `conversation_id` directly, without the existence
/// check of `add_message` (a write that raced the conversation's delete, or
/// a row from before the process started).
async fn insert_raw_message(db: &DbConnection, conversation_id: &str) {
    let m = Message::user(conversation_id.into(), "orphan".into());
    let _: Option<Message> = db.db().create(m.id.clone()).content(m).await.unwrap();
}

/// The recent orphan sweep deletes what is left for a deleted conversation
/// for two sweeps, through the index and without scanning the tables; a
/// conversation that still exists is left alone, and rows of one this
/// process never deleted need the full scan.
#[tokio::test]
async fn recent_orphan_sweep_follows_deleted_conversations() {
    let db = test_db().await;
    db.set_session_last_accessed("s1", chrono::Utc::now())
        .await
        .unwrap();
    let kept = db.create_conversation("s1", "kept").await.unwrap();
    db.add_message(&kept.id_string(), "user", "kept")
        .await
        .unwrap();
    let gone = db.create_conversation("s1", "gone").await.unwrap();
    let gone = gone.id_string();
    db.delete_conversation(&gone).await.unwrap();
    insert_raw_message(&db, &gone).await;
    insert_raw_message(&db, "from-before").await;

    assert_eq!(db.delete_recent_orphans().await.unwrap(), 1);
    insert_raw_message(&db, &gone).await; // lands after that sweep started
    assert_eq!(db.delete_recent_orphans().await.unwrap(), 1);
    insert_raw_message(&db, &gone).await;
    assert_eq!(db.delete_recent_orphans().await.unwrap(), 0); // forgotten

    db.note_conversation_deleted(&kept.id_string());
    assert_eq!(db.delete_recent_orphans().await.unwrap(), 0);
    assert_eq!(db.get_messages(&kept.id_string()).await.unwrap().len(), 1);

    // The full scan finds the rest: the row from before and the last one.
    assert_eq!(db.delete_orphans().await.unwrap(), 2);
    assert_eq!(db.get_messages(&kept.id_string()).await.unwrap().len(), 1);
}

/// More deleted conversations than are remembered: the next recent sweep
/// scans in full.
#[tokio::test]
async fn recent_orphan_sweep_scans_in_full_after_too_many_deletes() {
    let db = test_db().await;
    insert_raw_message(&db, "from-before").await;
    assert_eq!(db.delete_recent_orphans().await.unwrap(), 0);
    for i in 0..=super::connection::MAX_REMEMBERED_DELETES {
        db.note_conversation_deleted(&format!("c{i}"));
    }
    assert_eq!(db.delete_recent_orphans().await.unwrap(), 1);
    assert_eq!(db.delete_recent_orphans().await.unwrap(), 0);
}

/// Owner values around the sorted `known` ones that none of them equals:
/// the empty string, values before and after all of them, and for each a
/// prefix and two extensions (which sort right after it).
fn owners_around(known: &[String]) -> Vec<String> {
    let mut out = vec![String::new(), "0".to_string(), "zzzz".to_string()];
    for k in known {
        out.push(k[..k.len() / 2].to_string());
        out.push(format!("{k}0"));
        out.push(format!("{k}~"));
    }
    out.sort();
    out.dedup();
    out.retain(|o| !known.contains(o));
    out
}

/// Rows of `table`.
async fn rows_in(db: &DbConnection, table: &str) -> usize {
    #[derive(serde::Deserialize)]
    struct Count {
        count: usize,
    }
    let mut result = db
        .db()
        .query(format!("SELECT count() FROM {table} GROUP ALL"))
        .await
        .unwrap();
    let count: Option<Count> = result.take(0).unwrap();
    count.map_or(0, |c| c.count)
}

/// A tool call stored for `conversation_id` without any check.
async fn insert_raw_tool_call(db: &DbConnection, conversation_id: &str) {
    let t = ToolCall::new("m".into(), conversation_id.into(), "t".into(), "{}".into());
    let _: Option<ToolCall> = db.db().create(t.id.clone()).content(t).await.unwrap();
}

/// See [`full_orphan_sweep_finds_orphans_between_the_owners_that_exist`].
async fn check_orphans_between_owners(db: &DbConnection) {
    let now = chrono::Utc::now();
    let live: Vec<String> = ["m1", "s1", "s2"].map(String::from).to_vec();
    let mut conversations = Vec::new();
    for s in &live {
        db.set_session_last_accessed(s, now).await.unwrap();
    }
    // Every session, with a row or not, gets settings, dataset metadata and
    // a conversation with a message and a tool call.
    let rowless = owners_around(&live);
    for s in live.iter().chain(&rowless) {
        if !s.is_empty() {
            // The settings row's key is the session id.
            db.get_or_create_settings(s).await.unwrap();
        }
        let meta = super::models::DatasetMeta::new(
            s.clone(),
            "d".into(),
            "csv".into(),
            2,
            1,
            vec!["x".into()],
        );
        db.save_dataset_meta(&meta).await.unwrap();
        let id = db.create_conversation(s, "t").await.unwrap().id_string();
        let msg = db.add_message(&id, "user", "1.25").await.unwrap();
        let call = ToolCall::new(msg.id_string(), id.clone(), "t".into(), "{}".into());
        db.create_tool_call(&call).await.unwrap();
        if live.contains(s) {
            conversations.push(id);
        }
    }
    // More orphans in one gap than the sweep deletes per statement.
    for _ in 0..(2 * super::sessions::ORPHAN_BATCH + 3) {
        db.create_conversation("zzzz", "t").await.unwrap();
    }
    conversations.sort();
    let orphan_owners = owners_around(&conversations);
    for owner in &orphan_owners {
        insert_raw_message(db, owner).await;
        insert_raw_tool_call(db, owner).await;
    }
    for _ in 0..(2 * super::sessions::ORPHAN_BATCH + 5) {
        insert_raw_message(db, "zzzz").await;
    }

    let deleted = db.delete_orphans().await.unwrap();
    // The rowless sessions' conversations (their messages and tool calls go
    // with them uncounted), settings and dataset metadata, and the orphaned
    // messages and tool calls.
    let rowless_settings = rowless.iter().filter(|s| !s.is_empty()).count();
    let expected = rowless.len()
        + (2 * super::sessions::ORPHAN_BATCH + 3)
        + rowless_settings
        + rowless.len()
        + 2 * orphan_owners.len()
        + (2 * super::sessions::ORPHAN_BATCH + 5);
    assert_eq!(deleted, expected);
    for table in [
        "sessions",
        "settings",
        "dataset_meta",
        "conversations",
        "messages",
        "tool_calls",
    ] {
        assert_eq!(rows_in(db, table).await, live.len(), "{table}");
    }
    for id in &conversations {
        assert_eq!(db.get_messages(id).await.unwrap().len(), 1);
        let calls = db.get_tool_calls_for_conversation(id).await.unwrap();
        assert_eq!(calls.len(), 1);
    }
    for s in &live {
        assert_eq!(db.get_datasets_for_session(s).await.unwrap().len(), 1);
        let settings: Option<super::models::Settings> =
            db.db().select(("settings", s.as_str())).await.unwrap();
        assert!(settings.is_some(), "{s}");
    }
    assert_eq!(db.delete_orphans().await.unwrap(), 0);
}

/// The full sweep finds the orphans of every table wherever their owner
/// falls among the owners that exist (before, between and after them, a
/// prefix or an extension of one, the empty string), also more of them in
/// one gap than it deletes per statement, and keeps every row of an owner
/// that exists; on both engines.
#[tokio::test]
async fn full_orphan_sweep_finds_orphans_between_the_owners_that_exist() {
    check_orphans_between_owners(&test_db().await).await;
    let dir = tempfile::tempdir().unwrap();
    let db = DbConnection::connect_rocksdb(dir.path().join("db"))
        .await
        .unwrap();
    check_orphans_between_owners(&db).await;
}

#[tokio::test]
async fn session_rows_by_last_access() {
    let db = test_db().await;
    let now = chrono::Utc::now();
    db.set_session_last_accessed("old", now - chrono::Duration::hours(2))
        .await
        .unwrap(); // creates the row
    db.set_session_last_accessed("new", now).await.unwrap();
    let stale = db
        .session_ids_last_accessed_before(now - chrono::Duration::hours(1))
        .await
        .unwrap();
    assert_eq!(stale, vec!["old".to_string()]);
    let row = db.get_session("new").await.unwrap().unwrap();
    let drift = row.last_accessed_chrono().signed_duration_since(now);
    assert!(drift.num_milliseconds().abs() < 1, "{drift}");
}

/// The sweep's refresh updates an existing row and never creates one.
#[tokio::test]
async fn refreshing_last_access_never_creates_a_row() {
    let db = test_db().await;
    let at = chrono::Utc::now();
    assert!(!db.refresh_session_last_accessed("gone", at).await.unwrap());
    assert!(db.get_session("gone").await.unwrap().is_none());
    db.set_session_last_accessed("here", at - chrono::Duration::hours(1))
        .await
        .unwrap();
    assert!(db.refresh_session_last_accessed("here", at).await.unwrap());
    let row = db.get_session("here").await.unwrap().unwrap();
    let drift = row.last_accessed_chrono().signed_duration_since(at);
    assert!(drift.num_milliseconds().abs() < 1, "{drift}");
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

    let session =
        super::models::DbSession::new("test-session-1".to_string(), Some("user-123".to_string()));

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

    let session =
        super::models::DbSession::new("test-session-3".to_string(), Some("user-456".to_string()));

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
    let session_data = super::models::DbSession::new(
        "integration-test-session".to_string(),
        Some("test-user".to_string()),
    );
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
    let session1_data = super::models::DbSession::new(
        "isolation-session-1".to_string(),
        Some("user-1".to_string()),
    );
    let session1 = db
        .upsert_session(&session1_data)
        .await
        .expect("Failed to upsert session 1");

    let session2_data = super::models::DbSession::new(
        "isolation-session-2".to_string(),
        Some("user-2".to_string()),
    );
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
