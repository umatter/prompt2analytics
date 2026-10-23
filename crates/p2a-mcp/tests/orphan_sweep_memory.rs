//! What the orphan sweeps of the conversation store read and hold, measured
//! with a counting global allocator on a RocksDB store (the configuration of
//! a self-hosted server that keeps its history across restarts).
//!
//! The first sweep after every start (`DbConnection::delete_orphans`) used to
//! read every stored message, tool call and settings row in a grouped scan,
//! and to hold every conversation row (title included) at once, twice; the
//! periodic sweep (`delete_recent_orphans`) also held every conversation
//! row. A restart therefore raised the process memory by about the size of
//! the stored history. These tests hold the sweeps to reading only the rows
//! of owners that no longer exist (and the conversation rows, for their
//! ids), and to holding one session's conversation rows at a time.
//!
//! The allocator counts the Rust allocations of this test binary: every
//! value SurrealDB reads from the store is decoded there, while RocksDB's
//! own buffers and caches (C++) are not counted. The tests take turns
//! (`TURN`) for their whole run, setup included.
#![cfg(feature = "db")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use p2a_mcp::db::{DbConnection, Message, ToolCall};

/// The system allocator, counting the bytes it hands out and the most that
/// were live at once.
struct Counting;

static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

fn grew(n: usize) {
    let n = n as u64;
    ALLOCATED.fetch_add(n, Ordering::Relaxed);
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

fn shrank(n: usize) {
    LIVE.fetch_sub(n as u64, Ordering::Relaxed);
}

// SAFETY: every call is forwarded to `System` unchanged; the counters only
// observe the sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc` is passed on as is.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as for `alloc`.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grew(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        shrank(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            grew(new_size);
            shrank(layout.size());
        }
        new
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// One test measures at a time; held for the whole test.
static TURN: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// What `f` allocated in all, and the most it held at once above what was
/// live when it started.
struct Measured<T> {
    out: T,
    allocated: u64,
    peak: u64,
}

async fn measure<T>(f: impl std::future::Future<Output = T>) -> Measured<T> {
    let allocated = ALLOCATED.load(Ordering::Relaxed);
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    let out = f.await;
    Measured {
        out,
        allocated: ALLOCATED.load(Ordering::Relaxed) - allocated,
        peak: PEAK.load(Ordering::Relaxed).saturating_sub(live),
    }
}

/// A RocksDB store in a fresh directory (kept alive by the returned guard).
async fn rocksdb() -> (DbConnection, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = DbConnection::connect_rocksdb(dir.path().join("db"))
        .await
        .unwrap();
    (db, dir)
}

/// Text of `n` bytes that differs from row to row (`seed`), so that no layer
/// can share one copy between rows.
fn text(n: usize, seed: usize) -> String {
    let mut s = format!("{seed:08}-");
    let mut x = seed as u64 | 1;
    while s.len() < n {
        // xorshift: noisy, reproducible filler
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.push(char::from(b'a' + (x % 26) as u8));
    }
    s.truncate(n);
    s
}

/// A message stored for `conversation_id` without the existence check of
/// `add_message`: a row left over from before the process started.
async fn insert_orphan_message(db: &DbConnection, conversation_id: &str) {
    let m = Message::user(conversation_id.into(), "orphan".into());
    let _: Option<Message> = db.db().create(m.id.clone()).content(m).await.unwrap();
}

/// A tool call stored for `conversation_id` without any check.
async fn insert_orphan_tool_call(db: &DbConnection, conversation_id: &str) {
    let t = ToolCall::new("m".into(), conversation_id.into(), "t".into(), "{}".into());
    let _: Option<ToolCall> = db.db().create(t.id.clone()).content(t).await.unwrap();
}

/// The full sweep reads neither the messages and tool calls of conversations
/// that exist nor the settings of sessions that exist: it allocates less
/// than half of their bytes, where reading each of them once would allocate
/// all of them. It still finds the orphans before, between, after and next
/// to the conversations that exist.
#[tokio::test]
async fn full_orphan_sweep_does_not_read_the_rows_of_live_owners() {
    let _turn = TURN.lock().await;
    let (db, _dir) = rocksdb().await;
    let mut stored = 0usize;

    // Two sessions with long settings strings.
    for s in ["s1", "s2"] {
        db.set_session_last_accessed(s, chrono::Utc::now())
            .await
            .unwrap();
        let mut settings = db.get_or_create_settings(s).await.unwrap();
        settings.system_prompt = Some(text(2 * 1024 * 1024, s.len() + 7));
        settings.base_url = Some(text(2 * 1024 * 1024, s.len() + 11));
        stored += 4 * 1024 * 1024;
        db.update_settings(&settings).await.unwrap();
    }
    // Two conversations of s1 with 32 MiB of messages and 8 MiB of tool-call
    // arguments between them.
    let mut kept = Vec::new();
    for c in 0..2 {
        let id = db
            .create_conversation("s1", "chat")
            .await
            .unwrap()
            .id_string();
        for m in 0..32 {
            let msg = db
                .add_message(&id, "user", &text(512 * 1024, c * 100 + m))
                .await
                .unwrap();
            stored += msg.content.len();
            if m < 8 {
                let call = ToolCall::new(
                    msg.id_string(),
                    id.clone(),
                    "regression_ols".into(),
                    text(512 * 1024, 1_000 + c * 100 + m),
                );
                stored += call.arguments.len();
                db.create_tool_call(&call).await.unwrap();
            }
        }
        kept.push(id);
    }
    kept.sort();
    let orphan_owners = [
        String::new(),
        "0".to_string(),
        kept[0][..8].to_string(),
        format!("{}0", kept[0]),
        format!("{}~", kept[0]),
        format!("{}0", kept[1]),
        "zzz".to_string(),
    ];
    for owner in &orphan_owners {
        insert_orphan_message(&db, owner).await;
        insert_orphan_tool_call(&db, owner).await;
    }

    let sweep = measure(db.delete_orphans()).await;
    assert_eq!(sweep.out.unwrap(), 2 * orphan_owners.len());
    assert!(
        sweep.allocated < (stored / 2) as u64,
        "the sweep allocated {} bytes with {stored} bytes stored for live owners",
        sweep.allocated
    );
    for id in &kept {
        assert_eq!(db.get_messages(id).await.unwrap().len(), 32);
        let calls = db.get_tool_calls_for_conversation(id).await.unwrap();
        assert_eq!(calls.len(), 8);
    }
    for s in ["s1", "s2"] {
        let settings = db.get_or_create_settings(s).await.unwrap();
        assert_eq!(
            settings.system_prompt.map(|p| p.len()),
            Some(2 * 1024 * 1024)
        );
    }
}

/// Sessions with rows.
const SESSIONS: usize = 40;
/// Conversations of each session.
const PER_SESSION: usize = 5;

/// `SESSIONS` sessions of `PER_SESSION` conversations with titles of 128 KiB
/// (25 MiB of titles in all) and no messages, and, with `orphans`, that many
/// more of a session without a row. The bytes of the titles.
async fn store_with_long_titles(db: &DbConnection, orphans: usize) -> usize {
    let mut titles = 0usize;
    for s in 0..SESSIONS {
        let session = format!("s{s:02}");
        db.set_session_last_accessed(&session, chrono::Utc::now())
            .await
            .unwrap();
        for c in 0..PER_SESSION {
            let title = text(128 * 1024, 10_000 + s * 100 + c);
            titles += title.len();
            db.create_conversation(&session, &title).await.unwrap();
        }
    }
    for c in 0..orphans {
        db.create_conversation("no-row", &text(128 * 1024, 20_000 + c))
            .await
            .unwrap();
    }
    titles
}

/// The full sweep reads the conversation rows one session at a time: the
/// most it holds at once stays well below the bytes of the titles. The
/// conversations of a session without a row still go.
#[tokio::test]
async fn full_orphan_sweep_holds_one_sessions_conversation_rows_at_a_time() {
    let _turn = TURN.lock().await;
    let (db, _dir) = rocksdb().await;
    let titles = store_with_long_titles(&db, 4).await;

    let sweep = measure(db.delete_orphans()).await;
    assert_eq!(sweep.out.unwrap(), 4);
    assert!(
        sweep.peak < (titles / 8) as u64,
        "the sweep held {} bytes at once for {titles} bytes of titles",
        sweep.peak
    );
    for s in 0..SESSIONS {
        let left = db.list_conversations(&format!("s{s:02}")).await.unwrap();
        assert_eq!(left.len(), PER_SESSION);
    }
    assert!(db.list_conversations("no-row").await.unwrap().is_empty());
}

/// The periodic sweep does not hold every conversation row either: it looks
/// only at the conversations deleted since the sweep before the last one.
#[tokio::test]
async fn recent_orphan_sweep_does_not_hold_the_conversation_rows() {
    let _turn = TURN.lock().await;
    let (db, _dir) = rocksdb().await;
    let titles = store_with_long_titles(&db, 0).await;
    let gone = db
        .create_conversation("s00", "gone")
        .await
        .unwrap()
        .id_string();
    db.delete_conversation(&gone).await.unwrap();
    insert_orphan_message(&db, &gone).await;

    let sweep = measure(db.delete_recent_orphans()).await;
    assert_eq!(sweep.out.unwrap(), 1);
    assert!(
        sweep.peak < (titles / 8) as u64,
        "the sweep held {} bytes at once for {titles} bytes of titles",
        sweep.peak
    );
}
