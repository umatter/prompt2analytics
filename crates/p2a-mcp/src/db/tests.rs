//! Database layer tests
//!
//! Run with: `cargo test -p p2a-mcp --features full db_tests`

use std::collections::HashMap;

use surrealdb::error::Db as SurrealDbError;

use super::connection::{DbConnection, DbError, PersistenceLimits, failed_statement, truncate_to};
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

/// How far the session's byte counter is from what the store holds, as a
/// failure text (`None` when they agree).
async fn counter_gap(db: &DbConnection, session_id: &str) -> Option<String> {
    let stored = db.session_stored_bytes(session_id).await.unwrap();
    let counted = db.session_bytes_entry(session_id);
    (counted != Some(stored)).then(|| format!("counted {counted:?} bytes, stored {stored}"))
}

/// A content update counts only the bytes it adds over the content it
/// replaces, which is room for the new one: at the budget, a content of the
/// same size is kept whole, a shorter one frees the difference, and the
/// counter stays in step with the store. Before, the update counted the new
/// content on top of the old, so it was cut to the room left beside the old
/// content, and the counter stayed above the store by the old content's
/// bytes, refusing a message the session had room for.
#[tokio::test]
async fn a_content_update_counts_only_the_bytes_it_adds() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100),
        ..Default::default()
    })
    .await;
    let c = db.create_conversation("s1", "t").await.unwrap(); // 1
    let m = db
        .add_message(&c.id_string(), "assistant", &"a".repeat(60))
        .await
        .unwrap(); // 61
    let same = db
        .update_message_content(&m.id_string(), &"b".repeat(60))
        .await
        .unwrap();
    assert_eq!(same.content, "b".repeat(60));
    assert_eq!(counter_gap(&db, "s1").await, None);
    db.update_message_content(&m.id_string(), &"c".repeat(10))
        .await
        .unwrap(); // 11
    assert_eq!(counter_gap(&db, "s1").await, None);
    db.add_message(&c.id_string(), "user", &"d".repeat(89))
        .await
        .unwrap(); // exactly 100
    assert_eq!(counter_gap(&db, "s1").await, None);
}

/// A title change counts the bytes a longer title adds and releases those a
/// shorter one frees, so renaming back and forth keeps the session's byte
/// counter in step with the store, and the freed bytes are the session's
/// again. Before, a shorter title released nothing, and each long, short,
/// long cycle raised the counter by the difference until the session
/// refused writes it had room for.
#[tokio::test]
async fn a_shorter_title_releases_the_bytes_it_frees() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100),
        ..Default::default()
    })
    .await;
    let id = db
        .create_conversation("s1", &"t".repeat(40))
        .await
        .unwrap()
        .id_string();
    let mut problems = Vec::new();
    for (step, title) in [
        ("shorter", "u".to_string()),
        ("longer", "v".repeat(40)),
        ("shorter again", "w".to_string()),
        ("longer again", "x".repeat(40)),
        ("short", "y".to_string()),
    ] {
        match db.update_conversation_title(&id, &title).await {
            Ok(_) => {
                if let Some(gap) = counter_gap(&db, "s1").await {
                    problems.push(format!("{step}: {gap}"));
                }
            }
            Err(e) => problems.push(format!("{step}: {e}")),
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
    // The title holds 1 byte, so a message of 99 fills the budget exactly.
    db.add_message(&id, "user", &"m".repeat(99)).await.unwrap();
    assert_eq!(counter_gap(&db, "s1").await, None);
}

/// A write of an existing tool call counts only the bytes its texts add over
/// the stored texts they replace: an update that keeps the arguments does
/// not count them again, a result or error that replaces another counts the
/// difference, and near the budget the stored texts' bytes are room for
/// their replacements. Before, every update counted all the texts it wrote,
/// so the counter rose above the store with each one, and an update that
/// kept the arguments of a session at its budget cut them to nothing.
#[tokio::test]
async fn tool_call_writes_count_only_the_bytes_they_add() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(1000),
        ..Default::default()
    })
    .await;
    let c = db.create_conversation("s1", "t").await.unwrap();
    let m = db
        .add_message(&c.id_string(), "assistant", "")
        .await
        .unwrap();
    let call = ToolCall::new(
        m.id_string(),
        c.id_string(),
        "regression_ols".into(),
        "a".repeat(40),
    );
    let call = db.create_tool_call(&call).await.unwrap();
    let id = call.id_string();
    let mut problems = Vec::new();
    let mut step = async |what: &str| {
        if let Some(gap) = counter_gap(&db, "s1").await {
            problems.push(format!("{what}: {gap}"));
        }
    };
    step("create").await;

    let mut kept = call.clone();
    kept.result = Some("r".repeat(10));
    db.update_tool_call(&kept).await.unwrap();
    step("an update that keeps the arguments").await;
    db.complete_tool_call(&id, &"s".repeat(20), 3)
        .await
        .unwrap();
    step("a result that replaces another").await;
    db.fail_tool_call(&id, &"e".repeat(5), 3).await.unwrap();
    step("an error").await;
    db.mark_tool_call_running(&id).await.unwrap();
    step("a status change").await;
    let mut emptied = db.get_tool_call(&id).await.unwrap();
    (emptied.arguments, emptied.result, emptied.error) = ("b".repeat(10), None, None);
    db.update_tool_call(&emptied).await.unwrap();
    step("an update that drops the texts").await;
    assert!(problems.is_empty(), "{problems:#?}");

    // A session at its budget: the stored arguments are room for themselves.
    let full = limited_db(PersistenceLimits {
        session_db_bytes: Some(100),
        ..Default::default()
    })
    .await;
    let c = full.create_conversation("s1", "t").await.unwrap(); // 1
    let m = full
        .add_message(&c.id_string(), "assistant", "")
        .await
        .unwrap();
    let call = ToolCall::new(
        m.id_string(),
        c.id_string(),
        "regression_ols".into(),
        "x".repeat(99),
    );
    let mut call = full.create_tool_call(&call).await.unwrap(); // 100
    call.status = super::models::ToolCallStatus::Running;
    let updated = full.update_tool_call(&call).await.unwrap();
    assert_eq!(updated.arguments, "x".repeat(99));
    assert_eq!(counter_gap(&full, "s1").await, None);
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

/// `n` column names of `len` bytes each, distinct and in order: each
/// starts with its zero-padded index and `_`, which takes 7 bytes.
fn column_names(n: usize, len: usize) -> Vec<String> {
    assert!(len >= 7, "names shorter than 7 bytes are not distinct");
    (0..n)
        .map(|i| {
            format!("{i:06}_")
                .chars()
                .chain(std::iter::repeat('x'))
                .take(len)
                .collect()
        })
        .collect()
}

/// `kept` is the first `n` of `names` (compared without printing them).
fn assert_first(kept: &[String], names: &[String], n: usize) {
    assert_eq!(kept.len(), n, "names kept, of {}", names.len());
    assert!(kept == &names[..n], "the kept names are not the first {n}");
}

/// A dataset's metadata row keeps the first `MAX_STORED_COLUMN_NAMES`
/// column names and the dataset's full width in `column_count`, when it is
/// created and when it is replaced: the store's schema check of the row
/// takes time quadratic in the number of names, on the server's runtime.
#[tokio::test]
async fn dataset_meta_keeps_a_bounded_prefix_of_the_column_names() {
    use super::dataset_meta::MAX_STORED_COLUMN_NAMES as MAX;
    let db = test_db().await;
    let width = MAX + 700;
    let names = column_names(width, 9);
    let meta = |name: &str, names: Vec<String>| {
        let count = i32::try_from(names.len()).unwrap();
        super::models::DatasetMeta::new(
            "s1".into(),
            name.into(),
            "inline_csv".into(),
            2,
            count,
            names,
        )
    };

    let created = db
        .save_dataset_meta(&meta("wide", names.clone()))
        .await
        .unwrap();
    assert_first(&created.column_names, &names, MAX);
    assert_eq!(created.column_count, i32::try_from(width).unwrap());
    let stored = db.get_dataset_meta("s1", "wide").await.unwrap().unwrap();
    assert_first(&stored.column_names, &names, MAX);

    // Saving under the same name replaces the row, bounded the same way.
    let renamed: Vec<String> = names.iter().map(|n| format!("{n}_v2")).collect();
    let replaced = db
        .save_dataset_meta(&meta("wide", renamed.clone()))
        .await
        .unwrap();
    assert_eq!(replaced.id_string(), created.id_string());
    assert_first(&replaced.column_names, &renamed, MAX);
    assert_eq!(replaced.column_count, i32::try_from(width).unwrap());
    let rows = db.get_datasets_for_session("s1").await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_first(&rows[0].column_names, &renamed, MAX);

    // A dataset at the cap, and a narrow one, keep every name.
    let full = column_names(MAX, 9);
    let saved = db
        .save_dataset_meta(&meta("full", full.clone()))
        .await
        .unwrap();
    assert_first(&saved.column_names, &full, MAX);
    let narrow = column_names(3, 8);
    let saved = db
        .save_dataset_meta(&meta("narrow", narrow.clone()))
        .await
        .unwrap();
    assert_first(&saved.column_names, &narrow, 3);
}

/// The names a metadata row keeps are also bounded in bytes: the longest
/// prefix of the dataset's names within `MAX_STORED_COLUMN_NAME_BYTES`, so
/// a few long names do not make a large row either.
#[tokio::test]
async fn dataset_meta_bounds_the_bytes_of_the_column_names() {
    use super::dataset_meta::MAX_STORED_COLUMN_NAME_BYTES as MAX_BYTES;
    let db = test_db().await;
    let meta = |name: &str, names: Vec<String>| {
        let count = i32::try_from(names.len()).unwrap();
        super::models::DatasetMeta::new("s1".into(), name.into(), "csv".into(), 2, count, names)
    };
    // Names of 1 KiB: exactly `fit` of them fill the budget, one more passes it.
    let fit = MAX_BYTES / 1024;
    let names = column_names(fit + 6, 1024);
    let saved = db
        .save_dataset_meta(&meta("long", names.clone()))
        .await
        .unwrap();
    assert_first(&saved.column_names, &names, fit);
    assert_eq!(saved.column_count, i32::try_from(fit + 6).unwrap());

    // A first name longer than the whole budget leaves no names.
    let huge = vec!["y".repeat(MAX_BYTES + 1), "b".into()];
    let saved = db
        .save_dataset_meta(&meta("huge", huge.clone()))
        .await
        .unwrap();
    assert_first(&saved.column_names, &huge, 0);
    assert_eq!(saved.column_count, 2);
}

/// The metadata row of dataset `name` in session `s1`, with `n` column
/// names of 16 bytes: 128 + 2 + `name` + 3 + 16 `n` bytes against the
/// session's budget (457 for a 4-byte name and 20 names).
fn budget_meta(name: &str, n: usize) -> super::models::DatasetMeta {
    let count = i32::try_from(n).unwrap();
    super::models::DatasetMeta::new(
        "s1".into(),
        name.into(),
        "csv".into(),
        2,
        count,
        column_names(n, 16),
    )
}

/// A session's dataset metadata rows count against its `session_db_bytes`
/// budget, beside its conversation history: a row under a new name that
/// would pass the budget is refused and not written, a row replaced under
/// its name counts only the bytes it adds (refused, with the old row kept,
/// when they would pass the budget) and releases those it frees, and a
/// delete releases its rows' bytes; the counter stays in step with the
/// store. Before, the rows were not counted at all, and they outlive the
/// session's in-memory datasets (nothing deletes a row when its dataset
/// leaves the session's map), so the in-memory quotas did not bound them
/// either.
#[tokio::test]
async fn dataset_meta_rows_count_against_the_session_budget() {
    let budget = 4_000;
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(budget),
        ..Default::default()
    })
    .await;
    // 457 bytes a row (name "d000"): eight fit, a ninth does not.
    let mut stored = 0usize;
    let refused = loop {
        match db
            .save_dataset_meta(&budget_meta(&format!("d{stored:03}"), 20))
            .await
        {
            Ok(_) => stored += 1,
            Err(e) => break e,
        }
        assert!(stored <= 100, "rows past the budget");
    };
    assert!(matches!(refused, DbError::QuotaExceeded(_)), "{refused:?}");
    assert_eq!(stored, 8);
    let rows = db.get_datasets_for_session("s1").await.unwrap();
    assert_eq!(rows.len(), stored);
    assert_eq!(counter_gap(&db, "s1").await, None);
    assert_eq!(db.session_bytes_entry("s1"), Some(8 * 457));

    // The budget is one: a conversation title past it is refused too.
    let room = usize::try_from(budget - 8 * 457).unwrap();
    assert!(matches!(
        db.create_conversation("s1", &"t".repeat(room + 1)).await,
        Err(DbError::QuotaExceeded(_))
    ));

    // A row replaced by one of the same size fits at the budget; one that
    // grows past it is refused and the row stays as it was.
    db.save_dataset_meta(&budget_meta("d000", 20))
        .await
        .unwrap();
    let grown = db.save_dataset_meta(&budget_meta("d000", 60)).await;
    assert!(matches!(grown, Err(DbError::QuotaExceeded(_))), "{grown:?}");
    let kept = db.get_dataset_meta("s1", "d000").await.unwrap().unwrap();
    assert_eq!(kept.column_names.len(), 20);
    assert_eq!(counter_gap(&db, "s1").await, None);

    // A smaller replacement frees the difference.
    db.save_dataset_meta(&budget_meta("d000", 2)).await.unwrap();
    assert_eq!(db.session_bytes_entry("s1"), Some(8 * 457 - 18 * 16));
    assert_eq!(counter_gap(&db, "s1").await, None);

    // A delete frees its row: a new name fits again.
    db.delete_dataset_meta("s1", "d001").await.unwrap();
    assert_eq!(counter_gap(&db, "s1").await, None);
    db.save_dataset_meta(&budget_meta("new", 20)).await.unwrap();
    assert_eq!(counter_gap(&db, "s1").await, None);

    // A longer source path counts what it adds, a shorter one frees it.
    let path = "data/".repeat(10);
    db.update_dataset_source_path("s1", "d002", &path)
        .await
        .unwrap();
    assert_eq!(counter_gap(&db, "s1").await, None);
    let too_long = "p".repeat(budget as usize);
    assert!(matches!(
        db.update_dataset_source_path("s1", "d002", &too_long).await,
        Err(DbError::QuotaExceeded(_))
    ));
    db.update_dataset_source_path("s1", "d002", "a.csv")
        .await
        .unwrap();
    assert_eq!(counter_gap(&db, "s1").await, None);

    // Deleting them all frees everything.
    let deleted = db.delete_all_dataset_meta_for_session("s1").await.unwrap();
    assert_eq!(deleted, 8);
    assert_eq!(db.session_bytes_entry("s1"), Some(0));
    assert_eq!(counter_gap(&db, "s1").await, None);
    assert_eq!(db.store_turns_in_use(), 0);
}

/// Dataset metadata rows stored before the session's byte counter is
/// seeded (by a server without the budget, or before a restart) are part
/// of its seed: a new row past the budget beside them is refused.
#[tokio::test]
async fn stored_dataset_meta_rows_seed_the_session_counter() {
    let db = test_db().await;
    for i in 0..8 {
        db.save_dataset_meta(&budget_meta(&format!("d{i:03}"), 20))
            .await
            .unwrap();
    }
    let limited = db.clone().with_limits(PersistenceLimits {
        session_db_bytes: Some(4_000),
        ..Default::default()
    });
    assert_eq!(limited.session_stored_bytes("s1").await.unwrap(), 8 * 457);
    let refused = limited.save_dataset_meta(&budget_meta("d008", 20)).await;
    assert!(
        matches!(refused, Err(DbError::QuotaExceeded(_))),
        "{refused:?}"
    );
    assert_eq!(limited.session_bytes_entry("s1"), Some(8 * 457));
    assert_eq!(
        limited.get_datasets_for_session("s1").await.unwrap().len(),
        8
    );
}

/// Saves of one dataset name sent at the same time take turns: one row is
/// created and the others replace it, and the row is counted once. Before,
/// each save looked for the row before any had written it, and each
/// created one.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_saves_of_one_dataset_store_one_row() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100_000),
        ..Default::default()
    })
    .await;
    let saves: Vec<_> = (0..8)
        .map(|_| {
            let db = db.clone();
            tokio::spawn(async move { db.save_dataset_meta(&budget_meta("same", 20)).await })
        })
        .collect();
    for save in saves {
        save.await.unwrap().unwrap();
    }
    let rows = db.get_datasets_for_session("s1").await.unwrap();
    assert_eq!(rows.len(), 1, "rows of one dataset");
    assert_eq!(counter_gap(&db, "s1").await, None);
}

/// An orphan sweep that deletes the dataset metadata rows of a session
/// without a session row drops that session's byte counter, which counted
/// them; a session with a row keeps its rows and its counter.
#[tokio::test]
async fn an_orphan_sweep_of_dataset_meta_forgets_the_session_counter() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100_000),
        ..Default::default()
    })
    .await;
    db.upsert_session(&super::models::DbSession::new("s1".into(), None))
        .await
        .unwrap();
    db.save_dataset_meta(&budget_meta("kept", 20))
        .await
        .unwrap();
    let mut gone = budget_meta("gone", 20);
    gone.session_id = "gone".into();
    db.save_dataset_meta(&gone).await.unwrap();
    assert!(db.session_bytes_entry("gone").is_some());

    assert_eq!(db.delete_orphans().await.unwrap(), 1);
    assert_eq!(db.session_bytes_entry("gone"), None);
    assert!(
        db.get_datasets_for_session("gone")
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(db.get_datasets_for_session("s1").await.unwrap().len(), 1);
    assert_eq!(counter_gap(&db, "s1").await, None);
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

/// An add refused at the message cap brings the conversation's stored
/// `message_count` in step with its rows, in memory and on RocksDB. The add
/// that reached the cap can store its row and then fail its counter update
/// (here the store refuses the conversation row's update; a process that
/// stops between the two statements does the same), and every add after it
/// is refused before a recount: before, the conversation list kept showing
/// one message fewer than the refusal named until a clear or delete. A
/// refusal whose repair of the counter fails is still the refusal.
#[tokio::test]
async fn a_refused_add_brings_the_message_count_in_step() {
    for engine in ENGINES {
        let dir = tempfile::tempdir().unwrap();
        let limits = PersistenceLimits {
            messages_per_conversation: Some(3),
            ..Default::default()
        };
        let db = engine_db(engine, dir.path(), limits).await;
        let id = db.create_conversation("s1", "t").await.unwrap().id_string();
        db.add_message(&id, "user", "first").await.unwrap();
        db.add_message(&id, "assistant", "second").await.unwrap();
        refuse_conversation_writes(&db, Some("UPDATE")).await;
        // The add that reaches the cap stores its row; its counter update fails.
        let stored = db.add_message(&id, "user", "third").await;
        assert!(stored.is_err(), "{engine:?}: third add: {stored:?}");
        assert_eq!(listed_and_stored(&db, &id).await, (2, 3), "{engine:?}");
        let refused = db.add_message(&id, "user", "repair fails").await;
        assert!(
            matches!(refused, Err(DbError::QuotaExceeded(_))),
            "{engine:?}: add while the repair fails: {refused:?}"
        );

        refuse_conversation_writes(&db, None).await;
        let refused = db.add_message(&id, "user", "fourth").await;
        assert!(
            matches!(refused, Err(DbError::QuotaExceeded(_))),
            "{engine:?}: fourth add: {refused:?}"
        );
        assert_eq!(listed_and_stored(&db, &id).await, (3, 3), "{engine:?}");
        assert_eq!(db.store_turns_in_use(), 0);
    }
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

/// Make every `what` ('UPDATE' or 'DELETE') of a conversation row fail
/// inside the store, or (`None`) let them through again.
async fn refuse_conversation_writes(db: &DbConnection, what: Option<&str>) {
    let query = match what {
        Some(what) => format!(
            "DEFINE EVENT refuse ON TABLE conversations WHEN $event = '{what}' \
             THEN {{ THROW 'refused' }}"
        ),
        None => "REMOVE EVENT refuse ON TABLE conversations".to_string(),
    };
    db.db().query(query).await.unwrap().check().unwrap();
}

/// Conversation `id` (made by [`filled_conversation`]) still holds its 21
/// messages and its tool call, and its `message_count` says so.
async fn assert_conversation_whole(db: &DbConnection, id: &str) {
    assert_eq!(db.count_messages(id).await.unwrap(), 21);
    assert_eq!(rows_in(db, "tool_calls").await, 1);
    assert_eq!(db.get_conversation(id).await.unwrap().message_count, 21);
}

/// A store engine the transaction tests run on.
#[derive(Debug, Clone, Copy)]
enum Engine {
    /// In memory.
    Memory,
    /// RocksDB, whose rollback is an optimistic transaction's cancel.
    RocksDb,
}

/// The engines the rollback tests run on.
const ENGINES: [Engine; 2] = [Engine::Memory, Engine::RocksDb];

/// A store on `engine` (RocksDB in `dir`) with `limits`.
async fn engine_db(
    engine: Engine,
    dir: &std::path::Path,
    limits: PersistenceLimits,
) -> DbConnection {
    match engine {
        Engine::Memory => limited_db(limits).await,
        Engine::RocksDb => DbConnection::connect_rocksdb(dir.join("db"))
            .await
            .unwrap()
            .with_limits(limits),
    }
}

/// The error of a clear or delete that a store event refused names the
/// event's own message, not the "query was not executed" that the store
/// reports for the other statements of the transaction.
fn assert_refused(engine: Engine, error: DbError) {
    assert!(
        error.to_string().contains("refused"),
        "{engine:?}: the failing statement's error is reported: {error}"
    );
}

/// What [`failed_statement`] picks from the errors `make` builds, once for
/// each of several maps of them. `take_errors` returns a `HashMap`, and two
/// maps of the same entries iterate in different orders, so a pick that
/// depended on that order would differ between the maps.
fn failed_statements(make: &[(usize, fn() -> SurrealDbError)]) -> Vec<Option<surrealdb::Error>> {
    (0..16)
        .map(|_| {
            let errors: HashMap<usize, surrealdb::Error> = make
                .iter()
                .map(|(index, error)| (*index, surrealdb::Error::Db(error())))
                .collect();
            failed_statement(errors)
        })
        .collect()
}

/// A transaction's failed statement is told apart from the
/// `QueryNotExecuted` the store gives the statements before and after it,
/// whatever the order the map of errors iterates in.
#[test]
fn failed_statement_picks_the_cause_over_the_not_executed() {
    for picked in failed_statements(&[
        (0, || SurrealDbError::QueryNotExecuted),
        (1, || SurrealDbError::Thrown("refused".to_string())),
        (2, || SurrealDbError::QueryNotExecuted),
    ]) {
        assert!(
            matches!(
                &picked,
                Some(surrealdb::Error::Db(SurrealDbError::Thrown(message))) if message == "refused"
            ),
            "{picked:?}"
        );
    }
}

/// When no error is a plain `QueryNotExecuted` (a failed commit, which gives
/// each statement a `QueryNotExecutedDetail`; a transaction cancelled or
/// timed out between statements, which gives the statements that ran
/// `QueryCancelled` and the rest `QueryCancelled` or `QueryTimedout`), the
/// lowest index wins, so a timeout comes back as `QueryCancelled`.
#[test]
fn failed_statement_takes_the_lowest_index_among_causes() {
    for picked in failed_statements(&[
        (0, || SurrealDbError::QueryNotExecutedDetail {
            message: "a".to_string(),
        }),
        (1, || SurrealDbError::QueryNotExecutedDetail {
            message: "b".to_string(),
        }),
    ]) {
        assert!(
            matches!(
                &picked,
                Some(surrealdb::Error::Db(SurrealDbError::QueryNotExecutedDetail { message }))
                    if message == "a"
            ),
            "{picked:?}"
        );
    }
    for picked in failed_statements(&[
        (0, || SurrealDbError::QueryTimedout),
        (1, || SurrealDbError::QueryTimedout),
    ]) {
        assert!(
            matches!(
                picked,
                Some(surrealdb::Error::Db(SurrealDbError::QueryTimedout))
            ),
            "{picked:?}"
        );
    }
    // A timeout found after the first statement ran: that one is cancelled.
    for picked in failed_statements(&[
        (0, || SurrealDbError::QueryCancelled),
        (1, || SurrealDbError::QueryTimedout),
        (2, || SurrealDbError::QueryTimedout),
    ]) {
        assert!(
            matches!(
                picked,
                Some(surrealdb::Error::Db(SurrealDbError::QueryCancelled))
            ),
            "{picked:?}"
        );
    }
}

/// Only `QueryNotExecuted` errors still give an error (all alike, so which
/// index is returned cannot be seen in it), and no errors give none.
#[test]
fn failed_statement_with_no_cause_or_no_error() {
    for picked in failed_statements(&[
        (0, || SurrealDbError::QueryNotExecuted),
        (1, || SurrealDbError::QueryNotExecuted),
        (2, || SurrealDbError::QueryNotExecuted),
    ]) {
        assert!(
            matches!(
                picked,
                Some(surrealdb::Error::Db(SurrealDbError::QueryNotExecuted))
            ),
            "{picked:?}"
        );
    }
    assert!(failed_statements(&[]).iter().all(Option::is_none));
}

/// The layout [`failed_statement`] relies on, from the store itself: in a
/// `BEGIN`/`COMMIT` query of two counts with a failing statement between
/// them, both counts report `QueryNotExecuted`, so `Response::check` (the
/// first error by index; what the orphan sweep's range counts used) gives
/// that and loses the cause, while `failed_statement` gives the cause. In
/// memory and on RocksDB.
#[tokio::test]
async fn a_transactions_failed_statement_is_found_behind_not_executed() {
    for engine in ENGINES {
        let dir = tempfile::tempdir().unwrap();
        let db = engine_db(engine, dir.path(), PersistenceLimits::default()).await;
        let id = filled_conversation(&db, "s1", "kept").await;
        let transaction = || {
            db.db()
                .query("BEGIN TRANSACTION")
                .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
                .query("THROW 'refused'")
                .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
                .query("COMMIT TRANSACTION")
                .bind(("id", id.clone()))
        };

        let checked = transaction().await.unwrap().check().unwrap_err();
        assert!(
            matches!(
                checked,
                surrealdb::Error::Db(SurrealDbError::QueryNotExecuted)
            ),
            "{engine:?}: check: {checked:?}"
        );

        let mut response = transaction().await.unwrap();
        let errors = response.take_errors();
        let not_executed: Vec<usize> = errors
            .iter()
            .filter(|(_, error)| {
                matches!(
                    error,
                    surrealdb::Error::Db(SurrealDbError::QueryNotExecuted)
                )
            })
            .map(|(index, _)| *index)
            .collect();
        assert_eq!(
            (errors.len(), not_executed.len()),
            (3, 2),
            "{engine:?}: {errors:?}"
        );
        let picked = failed_statement(errors);
        assert!(
            matches!(
                &picked,
                Some(surrealdb::Error::Db(SurrealDbError::Thrown(message))) if message == "refused"
            ),
            "{engine:?}: failed_statement: {picked:?}"
        );
    }
}

/// A clear is one transaction. When its last step, the counter reset, fails
/// (standing in for the process stopping between the row deletes and the
/// reset), the rows are not deleted either, with and without the byte
/// budget, in memory and on RocksDB: before, the messages went and
/// `message_count` kept the old count. The error is the failed statement's.
#[tokio::test]
async fn a_clear_that_stops_part_way_deletes_nothing() {
    for (engine, session_db_bytes) in ENGINES
        .into_iter()
        .flat_map(|engine| [(engine, None), (engine, Some(100_000))])
    {
        let dir = tempfile::tempdir().unwrap();
        let limits = PersistenceLimits {
            session_db_bytes,
            ..Default::default()
        };
        let db = engine_db(engine, dir.path(), limits).await;
        let id = filled_conversation(&db, "s1", "kept").await;
        refuse_conversation_writes(&db, Some("UPDATE")).await;

        assert_refused(engine, db.clear_messages(&id).await.unwrap_err());
        assert_conversation_whole(&db, &id).await;
        if session_db_bytes.is_some() {
            assert_counter_matches_store(&db, "s1").await;
        }

        refuse_conversation_writes(&db, None).await;
        assert_eq!(db.clear_messages(&id).await.unwrap(), 21);
        assert_eq!(db.count_messages(&id).await.unwrap(), 0);
        assert_eq!(rows_in(&db, "tool_calls").await, 0);
        let conv = db.get_conversation(&id).await.unwrap();
        assert_eq!((conv.message_count, conv.last_message_preview), (0, None));
        if session_db_bytes.is_some() {
            assert_counter_matches_store(&db, "s1").await;
        }
        assert_eq!(db.store_turns_in_use(), 0);
    }
}

/// A conversation delete is one transaction. When its last step, deleting
/// the conversation row, fails, its messages and tool calls are not deleted
/// either, in memory and on RocksDB: before, they went and the conversation
/// stayed listed with its old `message_count`. The error is the failed
/// statement's.
#[tokio::test]
async fn a_delete_that_stops_part_way_deletes_nothing() {
    for (engine, session_db_bytes) in ENGINES
        .into_iter()
        .flat_map(|engine| [(engine, None), (engine, Some(100_000))])
    {
        let dir = tempfile::tempdir().unwrap();
        let limits = PersistenceLimits {
            session_db_bytes,
            ..Default::default()
        };
        let db = engine_db(engine, dir.path(), limits).await;
        let id = filled_conversation(&db, "s1", "kept").await;
        refuse_conversation_writes(&db, Some("DELETE")).await;

        assert_refused(engine, db.delete_conversation(&id).await.unwrap_err());
        assert_conversation_whole(&db, &id).await;
        if session_db_bytes.is_some() {
            assert_counter_matches_store(&db, "s1").await;
        }

        refuse_conversation_writes(&db, None).await;
        db.delete_conversation(&id).await.unwrap();
        assert!(matches!(
            db.get_conversation(&id).await,
            Err(DbError::NotFound(_))
        ));
        assert_eq!(rows_in(&db, "messages").await, 0);
        assert_eq!(rows_in(&db, "tool_calls").await, 0);
        if session_db_bytes.is_some() {
            assert_counter_matches_store(&db, "s1").await;
        }
        assert_eq!(db.store_turns_in_use(), 0);
    }
}

/// Conversation `id`'s `message_count` (0 once its row is deleted) and the
/// message rows it holds, read in one transaction, so both come from one
/// snapshot of the store.
async fn listed_and_stored(db: &DbConnection, id: &str) -> (i64, i64) {
    #[derive(serde::Deserialize)]
    struct CountRow {
        count: i64,
    }
    let mut response = db
        .db()
        .query("BEGIN TRANSACTION")
        .query("SELECT VALUE message_count FROM $conv")
        .query("SELECT count() FROM messages WHERE conversation_id = $id GROUP ALL")
        .query("COMMIT TRANSACTION")
        .bind(("conv", surrealdb::RecordId::from(("conversations", id))))
        .bind(("id", id.to_string()))
        .await
        .unwrap()
        .check()
        .unwrap();
    let listed: Vec<i64> = response.take(0).unwrap();
    let stored: Option<CountRow> = response.take(1).unwrap();
    (
        listed.first().copied().unwrap_or(0),
        stored.map_or(0, |row| row.count),
    )
}

/// Make each `event` ('CREATE', 'UPDATE' or 'DELETE') of a `table` row take
/// `ms` milliseconds longer (a store event sleeps), so that a write is still
/// running in the store when the test acts on it.
async fn slow_writes(db: &DbConnection, table: &str, event: &str, ms: u64) {
    db.db()
        .query(format!(
            "DEFINE EVENT slow_{} ON TABLE {table} WHEN $event = '{event}' \
             THEN {{ sleep({ms}ms) }}",
            event.to_lowercase()
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
}

/// Make each message delete take a while, so that a clear or delete of
/// [`filled_conversation`]'s 21 messages is still running in the store when
/// its caller is dropped.
async fn slow_message_deletes(db: &DbConnection) {
    slow_writes(db, "messages", "DELETE", 20).await;
}

/// Drop `call` (named `what` in the failure) while the store runs it: the
/// call must first hold store turn `key` (a conversation id, or
/// `session:<id>` for a creation) and then not have finished 100 ms later (a
/// [`slow_writes`] event makes it last longer). A call dropped before it
/// holds its turn does nothing, which would leave the dropped-part-way path
/// untested.
async fn drop_while_running(
    db: &DbConnection,
    key: &str,
    what: impl std::fmt::Debug,
    call: impl Future,
) {
    let mut call = std::pin::pin!(call);
    tokio::select! {
        _ = &mut call => panic!("{what:?} finished before it was seen holding its turn"),
        held = turn_is_held(db, key) => assert!(held, "{what:?} never held its turn"),
    }
    let dropped = tokio::time::timeout(std::time::Duration::from_millis(100), call).await;
    assert!(dropped.is_err(), "{what:?} finished before it was dropped");
}

/// What the dropped-call tests do to a conversation.
#[derive(Debug, Clone, Copy)]
enum Op {
    /// Clear its messages ([`DbConnection::clear_messages`]).
    Clear,
    /// Delete it ([`DbConnection::delete_conversation`]).
    Delete,
}

/// Start `op` on conversation `id` and drop the call while the store runs
/// it ([`slow_message_deletes`]).
async fn drop_part_way(db: &DbConnection, op: Op, id: &str) {
    let call = async {
        match op {
            Op::Clear => db.clear_messages(id).await.map(|_| ()),
            Op::Delete => db.delete_conversation(id).await,
        }
    };
    drop_while_running(db, id, op, call).await;
}

/// A clear or delete whose caller is dropped part way (an HTTP client that
/// disconnects; the store still finishes the query it was sent) leaves the
/// conversation's `message_count` in step with its rows and, with the byte
/// budget, the session's counter in step with the store, and gives up its
/// turn once the store is done. While the clear and delete were separate
/// statements, the call was dropped between the message delete and the
/// counter reset (or the row delete), and the conversation kept its old
/// count with no messages. While the call ran in its caller's future, the
/// transaction finished but the bytes it removed were never released, so
/// the counter stayed above the store (and the turn ended with the caller,
/// while the store still ran the transaction).
#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_clear_or_delete_leaves_the_count_in_step() {
    let mut out_of_step = Vec::new();
    for session_db_bytes in [None, Some(100_000)] {
        let db = limited_db(PersistenceLimits {
            session_db_bytes,
            ..Default::default()
        })
        .await;
        slow_message_deletes(&db).await;
        let pass = if session_db_bytes.is_some() {
            "budgeted"
        } else {
            "unbudgeted"
        };
        for op in [Op::Clear, Op::Delete] {
            let id = filled_conversation(&db, "s1", &format!("{op:?}")).await;
            drop_part_way(&db, op, &id).await;
            // Wait (a bounded number of polls) for the store to commit the
            // message delete, and then for the conversation's turn: the
            // call holds it until it has released the bytes it removed.
            for _ in 0..500 {
                if listed_and_stored(&db, &id).await.1 == 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            drop(db.conversation_turn(&id).await);
            let (listed, stored) = listed_and_stored(&db, &id).await;
            if listed != stored {
                out_of_step.push(format!("{pass} {op:?}: listed {listed}, stored {stored}"));
            }
            if session_db_bytes.is_some()
                && let Some(gap) = counter_gap(&db, "s1").await
            {
                out_of_step.push(format!("{pass} {op:?}: {gap}"));
            }
            // The dropped call's entry can outlast this test's own turn by
            // a moment (`StoreTurn::drop` lets the next call in first).
            assert!(
                turns_released(&db).await,
                "{pass} {op:?}: a turn stays in use"
            );
        }
    }
    assert!(out_of_step.is_empty(), "{out_of_step:?}");
}

/// A clear or delete sent again at once after its first call was dropped
/// (a client that disconnects and retries) waits for the first one, which
/// keeps the conversation's turn until the store has finished, and then
/// finds nothing: on RocksDB, which fails one of two conflicting
/// transactions, it does not fail on a write conflict. Before, the turn
/// ended with the dropped caller, the retry ran alongside the transaction
/// still in the store, and one of them failed.
#[tokio::test(flavor = "multi_thread")]
async fn a_retry_after_a_dropped_clear_or_delete_waits_for_it() {
    let dir = tempfile::tempdir().unwrap();
    let limits = PersistenceLimits {
        session_db_bytes: Some(100_000),
        ..Default::default()
    };
    let db = engine_db(Engine::RocksDb, dir.path(), limits).await;
    slow_message_deletes(&db).await;
    for op in [Op::Clear, Op::Delete] {
        let id = filled_conversation(&db, "s1", &format!("{op:?}")).await;
        drop_part_way(&db, op, &id).await;
        match op {
            Op::Clear => {
                let retried = db.clear_messages(&id).await;
                assert!(matches!(retried, Ok(0)), "clear retried: {retried:?}");
                let conv = db.get_conversation(&id).await.unwrap();
                assert_eq!((conv.message_count, conv.last_message_preview), (0, None));
            }
            Op::Delete => {
                let retried = db.delete_conversation(&id).await;
                assert!(
                    matches!(retried, Err(DbError::NotFound(_))),
                    "delete retried: {retried:?}"
                );
            }
        }
        assert_eq!(listed_and_stored(&db, &id).await, (0, 0), "{op:?}");
        assert_counter_matches_store(&db, "s1").await;
        // The dropped call's entry can outlast the retry by a moment
        // (`StoreTurn::drop` lets the next call in first).
        assert!(turns_released(&db).await, "{op:?}: a turn stays in use");
    }
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

/// Whether a call holds store turn `key`, not only waits for it (a call is
/// counted among [`turn_has_users`] before it has the lock), polled a
/// bounded number of times (no timing assertion: a slow machine only polls
/// longer).
async fn turn_is_held(db: &DbConnection, key: &str) -> bool {
    for _ in 0..500 {
        if db.store_turn_held(key) {
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

/// The writes of an existing tool-call row.
#[derive(Debug, Clone, Copy)]
enum ToolCallWrite {
    /// [`DbConnection::update_tool_call`].
    Update,
    /// [`DbConnection::mark_tool_call_running`].
    Running,
    /// [`DbConnection::complete_tool_call`].
    Complete,
    /// [`DbConnection::fail_tool_call`].
    Fail,
}

impl ToolCallWrite {
    const ALL: [ToolCallWrite; 4] = [Self::Update, Self::Running, Self::Complete, Self::Fail];

    /// Run this write on `call`.
    async fn run(self, db: &DbConnection, call: &ToolCall) -> Result<ToolCall, DbError> {
        let id = call.id_string();
        match self {
            Self::Update => {
                let mut done = call.clone();
                done.complete("updated".into());
                db.update_tool_call(&done).await
            }
            Self::Running => db.mark_tool_call_running(&id).await,
            Self::Complete => db.complete_tool_call(&id, "ok", 3).await,
            Self::Fail => db.fail_tool_call(&id, "failed", 3).await,
        }
    }

    /// Run this write on `call` in a task of its own.
    fn spawn(
        self,
        db: &DbConnection,
        call: &ToolCall,
    ) -> tokio::task::JoinHandle<Result<ToolCall, DbError>> {
        let (db, call) = (db.clone(), call.clone());
        tokio::spawn(async move { self.run(&db, &call).await })
    }
}

/// The one tool call of conversation `id` (made by [`filled_conversation`]).
async fn only_tool_call(db: &DbConnection, id: &str) -> ToolCall {
    let mut calls = db.get_tool_calls_for_conversation(id).await.unwrap();
    assert_eq!(calls.len(), 1);
    calls.remove(0)
}

/// A tool-call write sent while a clear of its conversation runs in the
/// store (a chat stream finishing a tool call while its client clears the
/// conversation) waits for the clear's turn and then finds the row gone
/// (`NotFound`, and nothing counted against the byte budget): on RocksDB,
/// which fails one of two conflicting transactions, the clear does not
/// fail. Before, the tool-call writes took no turn, the write committed on
/// a row the clear's transaction had deleted, and the clear failed whole on
/// a write conflict.
#[tokio::test(flavor = "multi_thread")]
async fn a_tool_call_write_during_a_clear_waits_for_it() {
    let mut problems = Vec::new();
    for session_db_bytes in [None, Some(100_000)] {
        let dir = tempfile::tempdir().unwrap();
        let limits = PersistenceLimits {
            session_db_bytes,
            ..Default::default()
        };
        let db = engine_db(Engine::RocksDb, dir.path(), limits).await;
        slow_message_deletes(&db).await;
        for write in ToolCallWrite::ALL {
            let case = format!("{write:?} (budget {session_db_bytes:?})");
            let id = filled_conversation(&db, "s1", &format!("{write:?}")).await;
            let call = only_tool_call(&db, &id).await;
            let clear = tokio::spawn({
                let (db, id) = (db.clone(), id.clone());
                async move { db.clear_messages(&id).await }
            });
            assert!(turn_is_held(&db, &id).await, "{case}");
            // Let the clear send its transaction, which then runs through 21
            // slow message deletes. What the write and the clear return does
            // not depend on this pause; without it, the write could commit
            // before the transaction starts and not conflict without its turn.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let written = write.spawn(&db, &call).await.unwrap();
            let cleared = clear.await.unwrap();
            if !matches!(cleared, Ok(21)) {
                problems.push(format!("{case}: clear: {cleared:?}"));
            }
            if !matches!(written, Err(DbError::NotFound(_))) {
                problems.push(format!("{case}: write: {written:?}"));
            }
            let left = db.get_tool_calls_for_conversation(&id).await.unwrap();
            if !left.is_empty() {
                problems.push(format!("{case}: {} tool calls left", left.len()));
            }
            if session_db_bytes.is_some()
                && let Some(gap) = counter_gap(&db, "s1").await
            {
                problems.push(format!("{case}: {gap}"));
            }
            assert_eq!(db.store_turns_in_use(), 0);
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

/// A tool-call write holds its conversation's turn until the store has
/// written it, also when its caller is dropped part way (a chat stream
/// whose client disconnects): a clear sent meanwhile waits for it, and both
/// succeed on RocksDB, with the byte counter in step with the store. Before,
/// the write took no turn, and the clear's transaction, which then ran
/// alongside it, failed on a write conflict when it committed.
#[tokio::test(flavor = "multi_thread")]
async fn a_clear_during_a_tool_call_write_waits_for_it() {
    let dir = tempfile::tempdir().unwrap();
    let limits = PersistenceLimits {
        session_db_bytes: Some(100_000),
        ..Default::default()
    };
    let db = engine_db(Engine::RocksDb, dir.path(), limits).await;
    // The clear's transaction outlasts the write's, so a write that ran
    // alongside it commits while it is still open.
    slow_writes(&db, "tool_calls", "UPDATE", 300).await;
    slow_message_deletes(&db).await;
    for write in ToolCallWrite::ALL {
        // Waited for.
        let id = filled_conversation(&db, "s1", &format!("{write:?}")).await;
        let call = only_tool_call(&db, &id).await;
        let written = write.spawn(&db, &call);
        assert!(
            turn_is_held(&db, &id).await,
            "{write:?} takes its conversation's turn"
        );
        let cleared = db.clear_messages(&id).await;
        let written = written.await.unwrap();
        assert!(written.is_ok(), "{write:?}: write: {written:?}");
        assert!(matches!(cleared, Ok(21)), "{write:?}: clear: {cleared:?}");
        assert_eq!(rows_in(&db, "tool_calls").await, 0, "{write:?}");
        assert_counter_matches_store(&db, "s1").await;
        assert_eq!(db.store_turns_in_use(), 0);

        // Dropped part way.
        let id = filled_conversation(&db, "s1", &format!("dropped {write:?}")).await;
        let call = only_tool_call(&db, &id).await;
        drop_while_running(&db, &id, write, write.run(&db, &call)).await;
        let cleared = db.clear_messages(&id).await;
        assert!(
            matches!(cleared, Ok(21)),
            "dropped {write:?}: clear: {cleared:?}"
        );
        assert_eq!(rows_in(&db, "tool_calls").await, 0, "dropped {write:?}");
        assert_counter_matches_store(&db, "s1").await;
        // The dropped write's entry can outlast the clear by a moment
        // (`StoreTurn::drop` lets the next call in first).
        assert!(
            turns_released(&db).await,
            "dropped {write:?}: a turn stays in use"
        );
    }
}

/// A tool-call create sent while a delete or clear of its conversation runs
/// in the store waits for that call's turn and is then `NotFound`: after a
/// delete the conversation is gone, after a clear the create's message is.
/// No tool call is left behind, neither for the orphan sweep nor, after a
/// clear, counted against the byte budget for a message nothing displays.
/// Before the create took the turn, it checked that the conversation existed
/// while the delete had not yet committed and wrote its row after the
/// delete's had gone; before it checked its message, a create after a clear
/// found the conversation and stored its row.
#[tokio::test(flavor = "multi_thread")]
async fn a_tool_call_create_during_a_clear_or_delete_leaves_no_row() {
    let mut problems = Vec::new();
    for engine in ENGINES {
        for op in [Op::Clear, Op::Delete] {
            let case = format!("{engine:?} {op:?}");
            let dir = tempfile::tempdir().unwrap();
            let limits = PersistenceLimits {
                session_db_bytes: Some(100_000),
                ..Default::default()
            };
            let db = engine_db(engine, dir.path(), limits).await;
            slow_message_deletes(&db).await;
            let id = filled_conversation(&db, "s1", "gone").await;
            let message_id = db.get_messages(&id).await.unwrap()[0].id_string();
            let running = tokio::spawn({
                let (db, id) = (db.clone(), id.clone());
                async move {
                    match op {
                        Op::Clear => db.clear_messages(&id).await.map(|_| ()),
                        Op::Delete => db.delete_conversation(&id).await,
                    }
                }
            });
            assert!(turn_is_held(&db, &id).await, "{case}");
            // As in a_tool_call_write_during_a_clear_waits_for_it.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let call = ToolCall::new(message_id, id.clone(), "regression_ols".into(), "{}".into());
            let created = db.create_tool_call(&call).await;
            running.await.unwrap().unwrap();
            if !matches!(created, Err(DbError::NotFound(_))) {
                problems.push(format!("{case}: create: {created:?}"));
            }
            let left = rows_in(&db, "tool_calls").await;
            if left != 0 {
                problems.push(format!("{case}: {left} tool calls left"));
            }
            if let Some(gap) = counter_gap(&db, "s1").await {
                problems.push(format!("{case}: {gap}"));
            }
            assert_eq!(db.store_turns_in_use(), 0);
        }
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

/// A tool call is created only for a message of its own conversation: one
/// whose message does not exist, or belongs to another conversation, is
/// `NotFound`, with no row stored and nothing counted.
#[tokio::test]
async fn a_tool_call_create_needs_its_message_in_its_conversation() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(1000),
        ..Default::default()
    })
    .await;
    let a = db.create_conversation("s1", "a").await.unwrap().id_string();
    let b = db.create_conversation("s1", "b").await.unwrap().id_string();
    let in_b = db
        .add_message(&b, "assistant", "")
        .await
        .unwrap()
        .id_string();
    for (case, message_id) in [
        ("a missing", "no-such-message".to_string()),
        ("another conversation's", in_b),
    ] {
        let call = ToolCall::new(message_id, a.clone(), "regression_ols".into(), "{}".into());
        let created = db.create_tool_call(&call).await;
        assert!(
            matches!(created, Err(DbError::NotFound(_))),
            "{case} message: {created:?}"
        );
    }
    assert_eq!(rows_in(&db, "tool_calls").await, 0);
    assert_eq!(counter_gap(&db, "s1").await, None);
}

/// A message add or save, or a conversation creation, whose caller is
/// dropped while the store writes its row (an HTTP client that disconnects)
/// keeps its turn until the store has written the row and, for a message,
/// the conversation's `message_count`: the same call sent at once after it,
/// one row below the cap, counts the dropped call's row and is refused, in
/// memory and on RocksDB. Before, the turn ended with the dropped caller,
/// the next call counted the rows before the dropped call's row was
/// committed, and both got in, one past the cap (and the dropped add's
/// `message_count` update never ran).
#[tokio::test(flavor = "multi_thread")]
async fn a_write_after_a_dropped_add_or_create_finds_the_cap_reached() {
    let mut outcomes = Vec::new();
    for engine in ENGINES {
        let dir = tempfile::tempdir().unwrap();
        let limits = PersistenceLimits {
            messages_per_conversation: Some(3),
            conversations_per_session: Some(3),
            ..Default::default()
        };
        let db = engine_db(engine, dir.path(), limits).await;
        let mut ids = Vec::new();
        for title in ["added", "saved"] {
            let id = db
                .create_conversation("s1", title)
                .await
                .unwrap()
                .id_string();
            db.add_message(&id, "user", "first").await.unwrap();
            db.add_message(&id, "assistant", "second").await.unwrap();
            ids.push(id);
        }
        slow_writes(&db, "messages", "CREATE", 200).await;
        slow_writes(&db, "conversations", "CREATE", 200).await;

        let added = &ids[0];
        drop_while_running(&db, added, "add", db.add_message(added, "user", "dropped")).await;
        let refused = db.add_message(added, "user", "refused").await.map(|_| ());
        let counts = listed_and_stored(&db, added).await;
        outcomes.push((engine, "add", refused, counts));

        let saved = &ids[1];
        let dropped = Message::user(saved.clone(), "dropped".into());
        drop_while_running(&db, saved, "save", db.save_message(&dropped)).await;
        let refused = db
            .save_message(&Message::user(saved.clone(), "refused".into()))
            .await
            .map(|_| ());
        let counts = listed_and_stored(&db, saved).await;
        outcomes.push((engine, "save", refused, counts));

        let create = db.create_conversation("s1", "dropped");
        drop_while_running(&db, "session:s1", "create", create).await;
        let refused = db.create_conversation("s1", "refused").await.map(|_| ());
        let stored = db.count_conversations("s1").await.unwrap() as i64;
        outcomes.push((engine, "create", refused, (stored, stored)));
        // A dropped call's task lets the next call in before its entry
        // leaves the map (`StoreTurn::drop`), so the entry can outlast the
        // next call by a moment.
        assert!(turns_released(&db).await, "{engine:?}: a turn stays in use");
    }
    // Left: the cases whose second call was not refused, or that do not
    // store (and list) three rows.
    outcomes.retain(|(_, _, refused, counts)| {
        !matches!(refused, Err(DbError::QuotaExceeded(_))) || *counts != (3, 3)
    });
    assert!(outcomes.is_empty(), "{outcomes:#?}");
}

/// How [`a_clear_after_a_dropped_write_waits_for_it`] changes a
/// conversation before its clear.
#[derive(Debug, Clone, Copy)]
enum RowWrite {
    /// [`DbConnection::update_conversation_title`].
    Title,
    /// [`DbConnection::set_conversation_archived`].
    Archive,
    /// [`DbConnection::update_message_content`].
    Content,
}

/// A title change, an archive or a content update whose caller is dropped
/// while the store writes it keeps the conversation's turn until the store
/// is done: a clear sent at once after it waits for it and succeeds on
/// RocksDB, the write lands, and the byte counter stays in step with the
/// store. Before, the turn ended with the dropped caller, and the clear's
/// transaction ran alongside the write still in the store and failed on a
/// write conflict when it committed.
#[tokio::test(flavor = "multi_thread")]
async fn a_clear_after_a_dropped_write_waits_for_it() {
    let dir = tempfile::tempdir().unwrap();
    let limits = PersistenceLimits {
        session_db_bytes: Some(100_000),
        ..Default::default()
    };
    let db = engine_db(Engine::RocksDb, dir.path(), limits).await;
    let writes = [RowWrite::Title, RowWrite::Archive, RowWrite::Content];
    let mut ids = Vec::new();
    for write in writes {
        ids.push(filled_conversation(&db, "s1", &format!("{write:?}")).await);
    }
    // The clear's own counter reset sleeps as well, so its transaction is
    // still open when a write that ran alongside it commits.
    slow_writes(&db, "conversations", "UPDATE", 200).await;
    slow_writes(&db, "messages", "UPDATE", 200).await;
    let mut problems = Vec::new();
    for (write, id) in writes.into_iter().zip(&ids) {
        // The content update fills the server's empty placeholder for its
        // answer, as a chat stream does.
        let placeholder = db.get_messages(id).await.unwrap();
        let placeholder = placeholder.iter().find(|m| m.content.is_empty()).unwrap();
        let message_id = placeholder.id_string();
        let call = async {
            match write {
                RowWrite::Title => db
                    .update_conversation_title(id, "renamed")
                    .await
                    .map(|_| ()),
                RowWrite::Archive => db.set_conversation_archived(id, true).await.map(|_| ()),
                RowWrite::Content => db
                    .update_message_content(&message_id, "a final answer")
                    .await
                    .map(|_| ()),
            }
        };
        drop_while_running(&db, id, write, call).await;
        let cleared = db.clear_messages(id).await;
        if !matches!(cleared, Ok(21)) {
            problems.push(format!("{write:?}: clear: {cleared:?}"));
        }
        let conv = db.get_conversation(id).await.unwrap();
        let landed = match write {
            RowWrite::Title => conv.title == "renamed",
            RowWrite::Archive => conv.is_archived,
            RowWrite::Content => true,
        };
        if !landed {
            problems.push(format!("{write:?}: the write was lost"));
        }
        let listed = (conv.message_count, conv.last_message_preview);
        let stored = db.count_messages(id).await.unwrap();
        if listed != (0, None) || stored != 0 {
            problems.push(format!("{write:?}: listed {listed:?}, stored {stored}"));
        }
        if let Some(gap) = counter_gap(&db, "s1").await {
            problems.push(format!("{write:?}: {gap}"));
        }
        // The dropped write's entry can outlast the clear by a moment
        // (`StoreTurn::drop` lets the next call in first).
        assert!(turns_released(&db).await, "{write:?}: a turn stays in use");
    }
    assert!(problems.is_empty(), "{problems:#?}");
}

/// A write sent while another call holds the store turn it needs, in
/// [`a_call_dropped_while_it_waits_for_its_turn_does_nothing`].
#[derive(Debug, Clone, Copy)]
enum Waiter {
    /// [`DbConnection::add_message`].
    Add,
    /// [`DbConnection::save_message`].
    Save,
    /// [`DbConnection::create_conversation`], which waits for its session's
    /// turn.
    Create,
    /// [`DbConnection::update_conversation_title`].
    Title,
    /// [`DbConnection::set_conversation_archived`].
    Archive,
    /// [`DbConnection::update_message_content`].
    Content,
    /// [`DbConnection::create_tool_call`].
    CreateToolCall,
    /// A write of the existing tool call.
    ToolCall(ToolCallWrite),
    /// [`DbConnection::clear_messages`].
    Clear,
    /// [`DbConnection::delete_conversation`].
    Delete,
}

impl Waiter {
    /// Every call that takes a turn, the clear and the delete last: a
    /// dropped call that still ran once its turn came would otherwise take
    /// the rows the later ones write.
    const ALL: [Waiter; 13] = [
        Self::Add,
        Self::Save,
        Self::Create,
        Self::Title,
        Self::Archive,
        Self::Content,
        Self::CreateToolCall,
        Self::ToolCall(ToolCallWrite::Update),
        Self::ToolCall(ToolCallWrite::Running),
        Self::ToolCall(ToolCallWrite::Complete),
        Self::ToolCall(ToolCallWrite::Fail),
        Self::Clear,
        Self::Delete,
    ];

    /// The store turn this call waits for, in conversation `id` of session
    /// `s1`.
    fn key(self, id: &str) -> String {
        match self {
            Self::Create => "session:s1".to_string(),
            _ => id.to_string(),
        }
    }

    /// Run this call on conversation `id` of session `s1` (made by
    /// [`filled_conversation`]), whose tool call is `call`.
    async fn run(self, db: &DbConnection, id: &str, call: &ToolCall) -> Result<(), DbError> {
        match self {
            Self::Add => db.add_message(id, "user", "waited").await.map(|_| ()),
            Self::Save => db
                .save_message(&Message::user(id.into(), "waited".into()))
                .await
                .map(|_| ()),
            Self::Create => db.create_conversation("s1", "waited").await.map(|_| ()),
            Self::Title => db.update_conversation_title(id, "waited").await.map(|_| ()),
            Self::Archive => db.set_conversation_archived(id, true).await.map(|_| ()),
            Self::Content => db
                .update_message_content(&call.message_id, "waited")
                .await
                .map(|_| ()),
            Self::CreateToolCall => {
                let new = ToolCall::new(
                    call.message_id.clone(),
                    id.into(),
                    "regression_ols".into(),
                    "{}".into(),
                );
                db.create_tool_call(&new).await.map(|_| ())
            }
            Self::ToolCall(write) => write.run(db, call).await.map(|_| ()),
            Self::Clear => db.clear_messages(id).await.map(|_| ()),
            Self::Delete => db.delete_conversation(id).await,
        }
    }
}

/// Session `s1`'s conversations with their messages and tool calls, and its
/// byte counter, as text to compare.
async fn session_state(db: &DbConnection) -> String {
    let mut state = Vec::new();
    for conv in db.list_conversations("s1").await.unwrap() {
        let id = conv.id_string();
        let messages = db.get_messages(&id).await.unwrap();
        let calls = db.get_tool_calls_for_conversation(&id).await.unwrap();
        state.push(format!("{conv:?}\n{messages:?}\n{calls:?}"));
    }
    format!("{state:#?}\ncounted {:?}", db.session_bytes_entry("s1"))
}

/// Whether every store turn has left the map, polled a bounded number of
/// times (no timing assertion: a slow machine only polls longer).
async fn turns_released(db: &DbConnection) -> bool {
    for _ in 0..500 {
        if db.store_turns_in_use() == 0 {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    false
}

/// A call dropped while it waits for its store turn (an HTTP client that
/// disconnects before its turn comes) does nothing, for every call that
/// takes a turn: once the turn is free, no row and no byte count has
/// changed. Before, each call waited for its turn inside the task it
/// spawned, which the caller's drop did not stop: the task kept the call's
/// payload while it waited and then ran the whole write.
#[tokio::test(flavor = "multi_thread")]
async fn a_call_dropped_while_it_waits_for_its_turn_does_nothing() {
    let db = limited_db(PersistenceLimits {
        session_db_bytes: Some(100_000),
        ..Default::default()
    })
    .await;
    let id = filled_conversation(&db, "s1", "kept").await;
    let call = only_tool_call(&db, &id).await;
    let mut problems = Vec::new();
    for waiter in Waiter::ALL {
        let before = session_state(&db).await;
        let key = waiter.key(&id);
        let held = match waiter {
            Waiter::Create => db.session_turn("s1").await,
            _ => db.conversation_turn(&id).await,
        };
        {
            let mut waiting = std::pin::pin!(waiter.run(&db, &id, &call));
            tokio::select! {
                done = &mut waiting => panic!("{waiter:?} finished while its turn was held: {done:?}"),
                queued = turn_has_users(&db, &key, 2) => {
                    assert!(queued, "{waiter:?} never waited for its turn");
                }
            }
            // It cannot finish while the turn is held; it is dropped waiting.
            let waited =
                tokio::time::timeout(std::time::Duration::from_millis(50), &mut waiting).await;
            assert!(
                waited.is_err(),
                "{waiter:?} finished while its turn was held"
            );
        }
        drop(held);
        assert!(turns_released(&db).await, "{waiter:?}: a turn stays in use");
        let after = session_state(&db).await;
        if after != before {
            problems.push(format!(
                "{waiter:?} changed the store:\n{before}\n->\n{after}"
            ));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n\n"));
}

/// `future` polled once, with a waker that does nothing: `Ready` when it
/// finished without waiting for anything.
fn poll_once<F: Future>(future: F) -> std::task::Poll<F::Output> {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    future.as_mut().poll(&mut cx)
}

/// A message add or save refused for its size, or because the server is low
/// on memory, is refused before it waits for the conversation's turn: while
/// another call holds the turn, the refusal comes at the first poll, and
/// the refused message never joins the turn's queue. Before, the size and
/// memory checks ran in the turn, so a refused message waited behind the
/// call holding it, its content kept until the refusal.
#[tokio::test]
async fn a_message_refused_for_size_or_memory_does_not_wait_for_its_turn() {
    use std::task::Poll;

    let db = limited_db(PersistenceLimits {
        message_bytes: Some(8),
        ..Default::default()
    })
    .await;
    let id = db.create_conversation("s1", "t").await.unwrap().id_string();
    let held = db.conversation_turn(&id).await;
    let too_large = Message::user(id.clone(), "123456789".into());
    let refused = [
        poll_once(db.add_message(&id, "user", "123456789")),
        poll_once(db.save_message(&too_large)),
    ];
    for refusal in &refused {
        assert!(
            matches!(refusal, Poll::Ready(Err(DbError::TooLarge(_)))),
            "{refusal:?}"
        );
    }
    if p2a_core::get_process_memory().is_some() {
        let low = db.clone().with_limits(PersistenceLimits {
            rss_soft_limit_bytes: Some(1),
            ..Default::default()
        });
        let refused = [
            poll_once(low.add_message(&id, "user", "x")),
            poll_once(low.save_message(&Message::user(id.clone(), "x".into()))),
        ];
        for refusal in &refused {
            assert!(
                matches!(refusal, Poll::Ready(Err(DbError::MemoryPressure))),
                "{refusal:?}"
            );
        }
    }
    assert_eq!(db.store_turn_users(&id), 1, "a refused message is queued");
    drop(held);
    assert!(turns_released(&db).await);
    assert_eq!(rows_in(&db, "messages").await, 0);
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
/// process never deleted need the full sweep.
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

    // The full sweep finds the rest: the row from before and the last one.
    assert_eq!(db.delete_orphans().await.unwrap(), 2);
    assert_eq!(db.get_messages(&kept.id_string()).await.unwrap().len(), 1);
}

/// More deleted conversations than are remembered: the next recent sweep
/// is a full one.
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

/// What `tracing` writes while a subscriber made with it is the thread's
/// default.
#[derive(Clone, Default)]
pub(super) struct Logs(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl Logs {
    /// A subscriber that writes WARN and above to these logs.
    pub(super) fn subscriber(&self) -> impl tracing::Subscriber + Send + Sync + use<> {
        share_callsite_interest();
        tracing_subscriber::fmt()
            .with_writer(self.clone())
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish()
    }

    pub(super) fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
}

impl std::io::Write for Logs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Logs {
    type Writer = Logs;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Register, once for the process, a dispatcher that is no thread's default
/// and wants no event, so that a [`Logs`] subscriber is never the only
/// registered dispatcher.
///
/// `tracing` caches a callsite's interest for the whole process when a
/// thread first reaches it. While just one dispatcher is registered
/// (tracing-core 0.1 `Dispatchers::rebuilder`), that interest comes from the
/// reaching thread's default dispatcher alone, so a callsite that another
/// test reached first on its own thread, with no subscriber, stayed disabled
/// for a `Logs` subscriber set on this thread, and its events were lost (now
/// and then, depending on which test got there first). With a second
/// dispatcher registered, a callsite's interest combines every registered
/// dispatcher's, under the lock that registering a dispatcher takes, and
/// registering the `Logs` subscriber rebuilds every callsite's interest.
/// This dispatcher answers never, so it adds nothing to that combination.
fn share_callsite_interest() {
    static OTHER: std::sync::OnceLock<tracing::Dispatch> = std::sync::OnceLock::new();
    OTHER.get_or_init(|| tracing::Dispatch::new(tracing::subscriber::NoSubscriber::new()));
}

/// Store `n` conversations of the rowless session "no-row" whose keys do not
/// survive `key_string` (the server writes UUID keys only), so that
/// `delete_conversation` cannot find them by the id the sweep reads.
async fn insert_odd_key_conversations(db: &DbConnection, n: usize) {
    use super::models::{Conversation, key_string};
    for i in 0..n {
        let key = format!("odd⟩{i:03}");
        let mut conv = Conversation::new("no-row".into(), "t".into());
        conv.id = surrealdb::RecordId::from(("conversations", key.as_str()));
        assert_ne!(key_string(&conv.id), key);
        let _: Option<Conversation> = db.db().create(conv.id.clone()).content(conv).await.unwrap();
    }
}

/// Run an orphan sweep (in full or of recent deletes) that must delete
/// nothing, and return the warnings it logged about conversations.
async fn sweep_warnings(db: &DbConnection, full: bool) -> Vec<String> {
    let logs = Logs::default();
    let deleted = {
        let _default = tracing::subscriber::set_default(logs.subscriber());
        if full {
            db.delete_orphans().await.unwrap()
        } else {
            db.delete_recent_orphans().await.unwrap()
        }
    };
    assert_eq!(deleted, 0);
    logs.text()
        .lines()
        .filter(|line| line.contains("WARN") && line.contains("conversations"))
        .map(String::from)
        .collect()
}

/// Conversations of a session without a row that `delete_conversation`
/// cannot find by the id the sweep reads for them are not read again batch
/// after batch until the batch bound: a full batch of them ends their gap
/// with a warning, and the rows stay. Every sweep reads them first again
/// and says so again, so the warning does not send anyone to wait for a
/// later sweep. A short batch of them is reported too.
#[tokio::test]
async fn full_orphan_sweep_stops_at_a_batch_it_cannot_delete() {
    use super::sessions::ORPHAN_BATCH;

    let db = test_db().await;
    insert_odd_key_conversations(&db, ORPHAN_BATCH + 3).await;
    for full in [true, false] {
        let warnings = sweep_warnings(&db, full).await;
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let w = &warnings[0];
        assert!(
            w.contains("could not delete") && w.contains("every sweep"),
            "{w}"
        );
        assert!(w.contains("by hand") && !w.contains("wait"), "{w}");
        let fields = format!("rows={} deleted=0", ORPHAN_BATCH + 3);
        assert!(w.contains(&fields), "{w}");
    }
    assert_eq!(rows_in(&db, "conversations").await, ORPHAN_BATCH + 3);

    let db = test_db().await;
    insert_odd_key_conversations(&db, 3).await;
    let warnings = sweep_warnings(&db, true).await;
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    let w = &warnings[0];
    assert!(w.contains("could not delete"), "{w}");
    assert!(w.contains("rows=3 deleted=0 left=3"), "{w}");
    assert_eq!(rows_in(&db, "conversations").await, 3);
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

/// The directories kept through a restart's purge are those of the stored
/// sessions `get_session` would still adopt: last accessed within the TTL,
/// by the same whole-minute rule.
#[tokio::test]
async fn keep_adoptable_session_dirs_keeps_the_sessions_within_the_ttl() {
    let db = test_db().await;
    let now = chrono::Utc::now();
    let id = || uuid::Uuid::new_v4().to_string();
    let (fresh, at_the_ttl, past) = (id(), id(), id());
    for (sid, minutes) in [(&fresh, 0), (&at_the_ttl, 60), (&past, 61)] {
        db.set_session_last_accessed(
            sid,
            now - chrono::Duration::minutes(minutes) - chrono::Duration::seconds(5),
        )
        .await
        .unwrap();
    }
    let mut kept = db.keep_adoptable_session_dirs(60).await.unwrap();
    kept.sort();
    let mut expected = vec![fresh, at_the_ttl];
    expected.sort();
    assert_eq!(kept, expected);
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
