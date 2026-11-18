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
    let id = gate
        .ensure(|| async {
            let id = api().create_session().await?;
            tracing::info!("Session created: {}", id);
            Ok(id)
        })
        .await?;
    if session.read().session_id.as_deref() != Some(id.as_str()) {
        session.write().set_session_id(id.clone());
    }
    Ok(id)
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
}
