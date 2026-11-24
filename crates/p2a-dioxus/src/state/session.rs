//! Session state management
//!
//! One backend session per tab (or desktop window), created when it is first
//! needed (a send, a data load, a new conversation) through one
//! single-flight [`SessionGate`], so concurrent first uses share one
//! creation.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use dioxus::prelude::*;
use futures::FutureExt;
use futures::future::{LocalBoxFuture, Shared};

use crate::api::api;

/// Information about a loaded dataset
#[derive(Clone, Debug)]
pub struct DatasetInfo {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
}

type Creation = Shared<LocalBoxFuture<'static, Result<String, String>>>;

enum GateState {
    /// No session known and none being created.
    Idle,
    /// A creation runs; every caller awaits this one. The number tells
    /// creations apart, so a late waiter of an old one changes nothing.
    Creating(u64, Creation),
    /// The session in use.
    Ready(String),
}

struct GateInner {
    state: GateState,
    generation: u64,
}

/// Single-flight session creation: while a creation runs, every
/// [`SessionGate::ensure`] awaits it instead of starting another; a known id
/// is returned at once; a failed creation is returned to every waiter and
/// the next call creates again. Cloning shares the gate.
#[derive(Clone)]
pub struct SessionGate(Rc<RefCell<GateInner>>);

impl Default for SessionGate {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionGate {
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(GateInner {
            state: GateState::Idle,
            generation: 0,
        })))
    }

    /// The session id, creating it with `create` unless one is known or
    /// being created.
    pub async fn ensure<F, Fut>(&self, create: F) -> Result<String, String>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<String, String>> + 'static,
    {
        let (generation, creation) = {
            let mut inner = self.0.borrow_mut();
            match &inner.state {
                GateState::Ready(id) => return Ok(id.clone()),
                GateState::Creating(generation, creation) => (*generation, creation.clone()),
                GateState::Idle => {
                    inner.generation += 1;
                    let generation = inner.generation;
                    let creation = create().boxed_local().shared();
                    inner.state = GateState::Creating(generation, creation.clone());
                    (generation, creation)
                }
            }
        };
        let result = creation.await;
        let mut inner = self.0.borrow_mut();
        if matches!(inner.state, GateState::Creating(g, _) if g == generation) {
            inner.state = match &result {
                Ok(id) => GateState::Ready(id.clone()),
                Err(_) => GateState::Idle,
            };
        }
        result
    }

    /// Drop `id` if it is the known session (it ended); a newer id stays.
    pub fn forget(&self, id: &str) {
        let mut inner = self.0.borrow_mut();
        if matches!(&inner.state, GateState::Ready(current) if current == id) {
            inner.state = GateState::Idle;
        }
    }

    /// The known session id, if any.
    pub fn current(&self) -> Option<String> {
        match &self.0.borrow().state {
            GateState::Ready(id) => Some(id.clone()),
            _ => None,
        }
    }
}

/// Session state for managing the backend session
#[derive(Clone)]
pub struct SessionState {
    /// Current session ID
    pub session_id: Option<String>,
    /// Whether the session has been initialized
    pub is_initialized: bool,
    /// Whether initialization is in progress
    pub is_loading: bool,
    /// Error message if initialization failed
    pub error: Option<String>,
    /// Currently loaded datasets
    pub loaded_datasets: Vec<DatasetInfo>,
    /// Counter to trigger dataset sidebar refresh (increment after tool execution)
    pub datasets_refresh_counter: u32,
    /// The one path that creates or restores this state's session
    /// ([`ensure_session`]); shared by clones.
    pub gate: SessionGate,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    /// Create a new session state
    pub fn new() -> Self {
        Self {
            session_id: None,
            is_initialized: false,
            is_loading: false,
            error: None,
            loaded_datasets: Vec::new(),
            datasets_refresh_counter: 0,
            gate: SessionGate::new(),
        }
    }

    /// Trigger a refresh of the datasets sidebar
    pub fn trigger_datasets_refresh(&mut self) {
        self.datasets_refresh_counter = self.datasets_refresh_counter.wrapping_add(1);
    }

    /// Get the current session ID if available
    pub fn get_session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Check if session is ready
    pub fn is_ready(&self) -> bool {
        self.is_initialized && self.session_id.is_some()
    }

    /// Set the session ID
    pub fn set_session_id(&mut self, id: String) {
        self.session_id = Some(id);
        self.is_initialized = true;
        self.error = None;
    }

    /// Set error state
    pub fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.is_loading = false;
    }

    /// Clear error state
    pub fn clear_error(&mut self) {
        self.error = None;
    }

    /// Set loading state
    pub fn set_loading(&mut self, loading: bool) {
        self.is_loading = loading;
    }

    /// Add a loaded dataset
    pub fn add_dataset(&mut self, name: String, rows: usize, cols: usize) {
        // Remove if already exists
        self.loaded_datasets.retain(|d| d.name != name);
        self.loaded_datasets.push(DatasetInfo { name, rows, cols });
    }

    /// Get loaded datasets
    pub fn get_datasets(&self) -> &[DatasetInfo] {
        &self.loaded_datasets
    }

    /// Clear all datasets
    pub fn clear_datasets(&mut self) {
        self.loaded_datasets.clear();
    }
}

/// The tab's session id, created on first need. Concurrent callers (a send
/// and a data load, a double click on `+`) share one creation through the
/// state's [`SessionGate`]. No signal borrow is held across the network
/// call: a guard kept alive there makes any concurrent access to the
/// session signal (a re-render reading `loaded_datasets`) panic.
pub async fn ensure_session(mut session: Signal<SessionState>) -> Result<String, String> {
    if let Some(id) = session.read().session_id.clone() {
        return Ok(id);
    }
    let gate = session.read().gate.clone();
    let id = gate.ensure(restore_or_create).await?;
    if session.read().session_id.as_deref() != Some(id.as_str()) {
        session.write().set_session_id(id.clone());
        remember_session(&id);
    }
    Ok(id)
}

/// What the app says when its session ended.
pub const SESSION_ENDED_TEXT: &str = "Your session ended (it was idle too long, or the server restarted). Its datasets and conversations are gone; load your data again. Retry sends your last message in a new session.";

/// Whether a server error means the session is gone: the codes
/// `session_expired` and `session_not_found`, `not_found` for a session
/// (the stream route also answers an unknown conversation with
/// `not_found`), and, from servers that send no code, the texts "Session
/// not found" / "Session expired".
pub fn session_lost(code: Option<&str>, message: &str) -> bool {
    let names_session =
        message.contains("Session not found") || message.contains("Session expired");
    match code {
        Some("session_expired" | "session_not_found") => true,
        Some("not_found") | None => names_session,
        Some(_) => false,
    }
}

/// [`session_lost`] for the error string a REST call returns
/// (`<message> (<code>)`, the `error_text` format).
pub fn session_lost_text(err: &str) -> bool {
    let split = err
        .strip_suffix(')')
        .and_then(|rest| rest.rsplit_once(" ("))
        .filter(|(_, code)| {
            !code.is_empty() && code.chars().all(|c| c.is_ascii_lowercase() || c == '_')
        });
    match split {
        Some((message, code)) => session_lost(Some(code), message),
        None => session_lost(None, err),
    }
}

/// Storage key of the native app's session id.
const SESSION_KEY: &str = "p2a-session";

/// Whether this build keeps its session id across restarts (desktop and
/// mobile; the web build never stores it).
const KEEPS_SESSION: bool = cfg!(any(feature = "desktop", feature = "mobile"));

/// Store `id` as the session to restore at the next start, when `native`.
pub fn remember_session_in<S: crate::platform::StorageBackend>(
    storage: &S,
    native: bool,
    id: &str,
) {
    if native && let Err(e) = storage.set(SESSION_KEY, &id) {
        tracing::warn!("Cannot store the session id: {}", e);
    }
}

/// The stored session id, if any.
pub fn stored_session_in<S: crate::platform::StorageBackend>(storage: &S) -> Option<String> {
    storage
        .get::<String>(SESSION_KEY)
        .ok()
        .filter(|id| !id.is_empty())
}

/// Remove the stored session id.
pub fn forget_stored_session_in<S: crate::platform::StorageBackend>(storage: &S) {
    if storage.contains_key(SESSION_KEY)
        && let Err(e) = storage.remove(SESSION_KEY)
    {
        tracing::warn!("Cannot remove the stored session id: {}", e);
    }
}

fn remember_session(id: &str) {
    if KEEPS_SESSION {
        remember_session_in(&crate::platform::create_storage(), true, id);
    }
}

fn forget_stored_session() {
    if KEEPS_SESSION {
        forget_stored_session_in(&crate::platform::create_storage());
    }
}

/// What follows a failed restore of the stored session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RestoreRefusal {
    /// The server says the session is gone: forget the stored id and create
    /// a new session.
    CreateNew,
    /// Any other failure (a server error, a capacity refusal, a timeout):
    /// keep the stored id, so the next use tries it again.
    KeepAndRetry,
}

/// What a failed restore with error `err` does next: only a session the
/// server says is gone is replaced; a transient failure must not cut the
/// link to the stored session's conversations.
pub(crate) fn restore_refusal(err: &str) -> RestoreRefusal {
    if session_lost_text(err) {
        RestoreRefusal::CreateNew
    } else {
        RestoreRefusal::KeepAndRetry
    }
}

/// The session to use when none is known: the stored one while the server
/// still has it (native builds), else a new one. A stored session the
/// server could not be asked about is kept and the error returned.
async fn restore_or_create() -> Result<String, String> {
    let client = api();
    let stored = if KEEPS_SESSION {
        stored_session_in(&crate::platform::create_storage())
    } else {
        None
    };
    if let Some(stored) = stored {
        match client.get_session(&stored).await {
            Ok(_) => {
                tracing::info!("Session restored: {}", stored);
                crate::api::set_current_session_id(Some(stored.clone()));
                return Ok(stored);
            }
            Err(e) => match restore_refusal(&e) {
                RestoreRefusal::CreateNew => {
                    tracing::info!("Stored session not restored ({e}); creating a new one");
                    forget_stored_session();
                }
                RestoreRefusal::KeepAndRetry => {
                    tracing::warn!("Stored session not reached ({e}); kept for the next try");
                    return Err(e);
                }
            },
        }
    }
    let id = client.create_session().await?;
    tracing::info!("Session created: {}", id);
    Ok(id)
}

/// Native start: restore the stored session through the gate (a send or
/// `+` during the restore waits for it instead of creating a second
/// session). The server adopts a stored session within its lifetime; one
/// it says is gone is forgotten and a new session is created; any other
/// failure keeps the stored id and the next use tries again. Without a
/// stored id nothing happens: the first use creates the session.
pub async fn restore_session(session: Signal<SessionState>) {
    if !KEEPS_SESSION || stored_session_in(&crate::platform::create_storage()).is_none() {
        return;
    }
    if let Err(e) = ensure_session(session).await {
        tracing::warn!("No session at start: {}", e);
    }
}

/// The session `old_id` ended: forget it everywhere (the gate, the state,
/// the session header, the stored id), drop its datasets and reset the
/// conversations and their live transcripts, so the next use creates a new
/// session.
pub fn on_session_lost(
    mut session: Signal<SessionState>,
    mut conv: Signal<super::ConversationState>,
    mut chat: Signal<super::ChatState>,
    old_id: &str,
) {
    let gate = session.read().gate.clone();
    gate.forget(old_id);
    let is_current = session.read().session_id.as_deref() == Some(old_id);
    if is_current {
        let mut s = session.write();
        s.session_id = None;
        s.is_initialized = false;
        s.clear_datasets();
        s.trigger_datasets_refresh();
        drop(s);
        crate::api::set_current_session_id(None);
        forget_stored_session();
        conv.write().reset();
        // The session's conversations went with it.
        chat.write().clear_live();
    }
}

/// Show a failed conversation call in the sidebar. When it says the session
/// ended, the session is forgotten and the conversations and their live
/// transcripts are reset first, so the next use creates a new session.
pub(crate) fn report_conversation_error(
    session: Signal<SessionState>,
    mut conv: Signal<super::ConversationState>,
    chat: Signal<super::ChatState>,
    e: String,
) {
    if session_lost_text(&e) {
        let old = session.read().session_id.clone();
        if let Some(old) = old {
            on_session_lost(session, conv, chat, &old);
        }
        conv.write().set_error(Some(SESSION_ENDED_TEXT.to_string()));
    } else {
        conv.write().set_error(Some(e));
    }
}

/// Create a conversation and make it current (the sidebar's "+" and
/// Ctrl+N; the caller has checked `switch_blocked`). A conversation created
/// while a send started meanwhile is listed, not made current. Failures go
/// through [`report_conversation_error`].
pub(crate) async fn start_new_conversation(
    session: Signal<SessionState>,
    mut conv: Signal<super::ConversationState>,
    chat: Signal<super::ChatState>,
) {
    conv.write().set_operating(true);
    let session_id = match ensure_session(session).await {
        Ok(id) => id,
        Err(e) => {
            report_conversation_error(session, conv, chat, e);
            return;
        }
    };
    match api()
        .create_conversation(&session_id, "New Conversation")
        .await
    {
        Ok(conversation) => {
            tracing::info!("Created conversation: {}", conversation.id);
            let id = conversation.id.clone();
            let mut c = conv.write();
            c.add_conversation(conversation);
            if !chat.read().is_busy() {
                c.set_current_conversation(Some(id));
            }
            c.set_operating(false);
        }
        Err(e) => {
            tracing::warn!("Failed to create conversation: {}", e);
            report_conversation_error(session, conv, chat, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::channel::oneshot;
    use futures::executor::LocalPool;
    use futures::task::LocalSpawnExt;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    #[test]
    fn session_lost_recognises_the_servers_answers() {
        assert!(session_lost(Some("session_expired"), "Session expired"));
        assert!(session_lost(Some("session_not_found"), "Session not found"));
        assert!(session_lost(Some("not_found"), "Session not found"));
        // The stream route answers an unknown conversation with not_found too.
        assert!(!session_lost(Some("not_found"), "Conversation not found"));
        // Older servers send no code.
        assert!(session_lost(None, "Session expired"));
        assert!(session_lost(None, "Session not found: 3f2a"));
        assert!(!session_lost(
            Some("capacity"),
            "Maximum number of sessions reached"
        ));
        assert!(!session_lost(None, "HTTP error: 404"));
    }

    #[test]
    fn session_lost_text_reads_the_rest_error_string() {
        use crate::api::error_text;
        assert!(session_lost_text(&error_text(
            "Session expired",
            Some("session_expired")
        )));
        assert!(session_lost_text(&error_text(
            "Session not found",
            Some("not_found")
        )));
        assert!(!session_lost_text(&error_text(
            "Conversation not found",
            Some("not_found")
        )));
        assert!(!session_lost_text(&error_text(
            "Maximum number of sessions reached",
            Some("capacity")
        )));
        assert!(!session_lost_text("HTTP error: 404"));
        assert!(session_lost_text("Session expired"));
    }

    /// Storage held in memory, standing in for the native config directory.
    #[derive(Default)]
    struct MemoryStorage(RefCell<std::collections::HashMap<String, String>>);

    impl crate::platform::StorageBackend for MemoryStorage {
        fn get<T: serde::de::DeserializeOwned>(
            &self,
            key: &str,
        ) -> Result<T, crate::platform::StorageError> {
            let map = self.0.borrow();
            let raw = map
                .get(key)
                .ok_or_else(|| crate::platform::StorageError(format!("no {key}")))?;
            serde_json::from_str(raw).map_err(|e| crate::platform::StorageError(e.to_string()))
        }

        fn set<T: serde::Serialize>(
            &self,
            key: &str,
            value: &T,
        ) -> Result<(), crate::platform::StorageError> {
            let raw = serde_json::to_string(value)
                .map_err(|e| crate::platform::StorageError(e.to_string()))?;
            self.0.borrow_mut().insert(key.to_string(), raw);
            Ok(())
        }

        fn remove(&self, key: &str) -> Result<(), crate::platform::StorageError> {
            self.0.borrow_mut().remove(key);
            Ok(())
        }

        fn contains_key(&self, key: &str) -> bool {
            self.0.borrow().contains_key(key)
        }
    }

    #[test]
    fn stored_session_round_trip() {
        let storage = MemoryStorage::default();
        assert_eq!(stored_session_in(&storage), None);
        remember_session_in(&storage, true, "s-1");
        assert!(storage.0.borrow().contains_key("p2a-session"));
        assert_eq!(stored_session_in(&storage).as_deref(), Some("s-1"));
        forget_stored_session_in(&storage);
        assert_eq!(stored_session_in(&storage), None);

        // The web build never stores the id.
        let web = MemoryStorage::default();
        remember_session_in(&web, false, "s-1");
        assert!(web.0.borrow().is_empty());
        assert_eq!(stored_session_in(&web), None);
    }

    #[test]
    fn gate_runs_one_creation_for_concurrent_callers() {
        let mut pool = LocalPool::new();
        let gate = SessionGate::new();
        let calls = Rc::new(Cell::new(0));
        let (tx, rx) = oneshot::channel::<Result<String, String>>();
        let rx = Rc::new(RefCell::new(Some(rx)));
        let got = Rc::new(RefCell::new(Vec::new()));
        for _ in 0..2 {
            let gate = gate.clone();
            let calls = calls.clone();
            let rx = rx.clone();
            let got = got.clone();
            pool.spawner()
                .spawn_local(async move {
                    let id = gate
                        .ensure(move || {
                            calls.set(calls.get() + 1);
                            let rx = rx.borrow_mut().take().expect("one creation");
                            async move { rx.await.unwrap_or_else(|_| Err("dropped".into())) }
                        })
                        .await;
                    got.borrow_mut().push(id);
                })
                .unwrap();
        }
        pool.run_until_stalled();
        assert!(got.borrow().is_empty(), "both wait for the creation");
        tx.send(Ok("s-1".to_string())).unwrap();
        pool.run();
        assert_eq!(calls.get(), 1);
        assert_eq!(
            *got.borrow(),
            [Ok("s-1".to_string()), Ok("s-1".to_string())]
        );
        assert_eq!(gate.current().as_deref(), Some("s-1"));
    }

    #[test]
    fn gate_reuses_a_known_id() {
        let gate = SessionGate::new();
        let calls = Rc::new(Cell::new(0));
        for _ in 0..3 {
            let calls = calls.clone();
            let id = futures::executor::block_on(gate.ensure(move || {
                calls.set(calls.get() + 1);
                async { Ok("s-1".to_string()) }
            }));
            assert_eq!(id.as_deref(), Ok("s-1"));
        }
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn gate_retries_after_a_failed_creation() {
        let mut pool = LocalPool::new();
        let gate = SessionGate::new();
        let (tx, rx) = oneshot::channel::<Result<String, String>>();
        let rx = Rc::new(RefCell::new(Some(rx)));
        let got = Rc::new(RefCell::new(Vec::new()));
        for _ in 0..2 {
            let (gate, rx, got) = (gate.clone(), rx.clone(), got.clone());
            pool.spawner()
                .spawn_local(async move {
                    let r = gate
                        .ensure(move || {
                            let rx = rx.borrow_mut().take().expect("one creation");
                            async move { rx.await.unwrap_or_else(|_| Err("dropped".into())) }
                        })
                        .await;
                    got.borrow_mut().push(r);
                })
                .unwrap();
        }
        pool.run_until_stalled();
        tx.send(Err("capacity".to_string())).unwrap();
        pool.run();
        assert_eq!(
            *got.borrow(),
            [Err("capacity".to_string()), Err("capacity".to_string())]
        );
        assert_eq!(gate.current(), None);
        let again = futures::executor::block_on(gate.ensure(|| async { Ok("s-2".to_string()) }));
        assert_eq!(again.as_deref(), Ok("s-2"));
    }

    #[test]
    fn forget_only_drops_the_named_id() {
        let gate = SessionGate::new();
        let _ = futures::executor::block_on(gate.ensure(|| async { Ok("old".to_string()) }));
        gate.forget("old");
        assert_eq!(gate.current(), None);
        let _ = futures::executor::block_on(gate.ensure(|| async { Ok("new".to_string()) }));
        // A second failure of the dead session must not drop its replacement.
        gate.forget("old");
        assert_eq!(gate.current().as_deref(), Some("new"));
    }

    #[test]
    fn only_a_session_the_server_says_is_gone_is_replaced_at_restore() {
        use crate::api::error_text;
        for gone in [
            error_text("Session expired", Some("session_expired")),
            error_text("Session not found", Some("session_not_found")),
            error_text("Session not found", Some("not_found")),
        ] {
            assert_eq!(restore_refusal(&gone), RestoreRefusal::CreateNew, "{gone}");
        }
        for transient in [
            error_text("Internal server error", Some("internal")),
            error_text("Maximum number of sessions reached", Some("capacity")),
            "HTTP error: 503".to_string(),
            "Request timed out".to_string(),
        ] {
            assert_eq!(
                restore_refusal(&transient),
                RestoreRefusal::KeepAndRetry,
                "{transient}"
            );
        }
    }

    fn conversation(id: &str) -> crate::api::Conversation {
        crate::api::Conversation {
            id: id.into(),
            session_id: "s1".into(),
            title: "Analysis".into(),
            created_at: "2026-11-18T10:00:00Z".into(),
            updated_at: "2026-11-18T10:00:00Z".into(),
            is_archived: false,
            message_count: 1,
            last_message_preview: None,
        }
    }

    /// Runs `f` with the signals of a session `s1` with one conversation
    /// `c1`, left with a live transcript.
    fn with_session_signals(
        f: impl FnOnce(
            Signal<SessionState>,
            Signal<super::super::ConversationState>,
            Signal<super::super::ChatState>,
        ),
    ) {
        let mut dom = VirtualDom::new(|| rsx! {});
        dom.rebuild_in_place();
        dom.in_scope(ScopeId::ROOT, || {
            let mut state = SessionState::new();
            state.set_session_id("s1".into());
            let session = Signal::new(state);
            let mut conv_state = super::super::ConversationState::new();
            conv_state.add_conversation(conversation("c1"));
            let conv = Signal::new(conv_state);
            let mut chat_state = super::super::ChatState::new();
            chat_state.add_user_message("q");
            chat_state.stash_live("c1");
            let chat = Signal::new(chat_state);
            f(session, conv, chat);
        });
    }

    #[test]
    fn a_conversation_error_saying_the_session_ended_forgets_it() {
        with_session_signals(|session, conv, chat| {
            let e = crate::api::error_text("Session expired", Some("session_expired"));
            report_conversation_error(session, conv, chat, e);
            assert_eq!(session.read().session_id, None);
            assert!(conv.read().conversations.is_empty());
            assert_eq!(conv.read().error.as_deref(), Some(SESSION_ENDED_TEXT));
            assert!(!conv.read().is_operating);
            assert!(!chat.read().has_live("c1"), "the live transcripts went too");
        });
    }

    #[test]
    fn a_lost_session_takes_its_live_transcripts() {
        with_session_signals(|session, conv, chat| {
            on_session_lost(session, conv, chat, "s1");
            assert_eq!(session.read().session_id, None);
            assert!(conv.read().conversations.is_empty());
            assert!(!chat.read().has_live("c1"), "the live transcripts went too");
        });
    }

    #[test]
    fn an_older_lost_session_keeps_the_current_live_transcripts() {
        with_session_signals(|session, conv, chat| {
            on_session_lost(session, conv, chat, "s0");
            assert_eq!(session.read().session_id.as_deref(), Some("s1"));
            assert!(chat.read().has_live("c1"));
        });
    }

    #[test]
    fn another_conversation_error_keeps_the_session() {
        with_session_signals(|session, mut conv, chat| {
            conv.write().set_operating(true);
            report_conversation_error(session, conv, chat, "HTTP error: 503".into());
            assert_eq!(session.read().session_id.as_deref(), Some("s1"));
            assert_eq!(conv.read().conversations.len(), 1);
            assert_eq!(conv.read().error.as_deref(), Some("HTTP error: 503"));
            assert!(!conv.read().is_operating);
            assert!(chat.read().has_live("c1"));
        });
    }
}

/// Asks the chat panel to open the add-data dialog: the dataset sidebar's
/// "Add data" button bumps the count, the chat panel (which owns the
/// dialog) opens it on every change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AddDataRequest(pub Signal<u64>);

/// Asks the chat panel to open the settings dialog on its model tab, the
/// same way: the welcome screen's "Connect a model" bumps the count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConnectModelRequest(pub Signal<u64>);
