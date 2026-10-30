//! Session management for multi-user HTTP transport.
//!
//! Each session maintains isolated dataset storage and state,
//! allowing multiple users to work independently.

use chrono::{DateTime, Utc};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, OnceLock, PoisonError, Weak};
use tokio::sync::{RwLock, Semaphore, broadcast};
use uuid::Uuid;

use p2a_core::data::{CleaningSession, Dataset, OperationRecord};
use p2a_core::polars::prelude::{
    CategoricalMapping, ChunkedArray, DataFrame, DataType, FrozenCategories, PolarsNumericType,
    PolarsResult, Series,
};
use p2a_core::{MemoryProfiler, SpatialWeights};

use crate::config::SessionConfig;

/// A unique session identifier.
pub type SessionId = String;

/// Session state for a single user connection.
pub struct Session {
    /// Unique session identifier
    pub id: SessionId,
    /// When the session was created
    pub created_at: DateTime<Utc>,
    /// When the session was last accessed
    pub last_accessed: RwLock<DateTime<Utc>>,
    /// Datasets loaded in this session
    pub datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    /// Global random seed for ML reproducibility
    pub global_seed: Arc<RwLock<Option<u64>>>,
    /// Cleaning sessions started with `cleaning_session_start`, keyed by
    /// cleaning-session id. Each is reference-counted so that a quota
    /// snapshot ([`Session::snapshot_state`]) shares it; a tool changes one
    /// through `Arc::make_mut`, which copies it only while a snapshot holds
    /// it.
    pub cleaning_sessions: Arc<RwLock<HashMap<String, Arc<CleaningSession>>>>,
    /// Spatial weights stored by `spatial_neighbors`, keyed by weights name.
    pub spatial_weights: Arc<RwLock<HashMap<String, SpatialWeights>>>,
    /// The profile behind `server_stats`' "Recent Memory History"
    /// (`include_history`) for this session. No quota charges it; instead
    /// its names are pruned to the datasets the session holds once a
    /// governed call that changed the state is kept (`Quotas::enforce`) or
    /// undone ([`restore_state`](Self::restore_state)) when a quota applies,
    /// and at every charge refresh
    /// ([`refresh_data_bytes`](Self::refresh_data_bytes)). Without a quota
    /// nothing prunes it, so the history can count datasets the session no
    /// longer holds.
    pub memory_profiler: Arc<RwLock<MemoryProfiler>>,
    /// Optional user ID (when authentication is enabled)
    pub user_id: Option<String>,
    /// Per-session execution permits, created on first use.
    exec_slots: OnceLock<Arc<Semaphore>>,
    /// Tool calls currently running in this session.
    active_calls: AtomicUsize,
    /// Cached in-memory size of the session's state (`datasets`,
    /// `cleaning_sessions`, `spatial_weights`), in bytes.
    data_bytes: AtomicU64,
    /// The shared-mapping part of `data_bytes`, cached with it.
    shared_labels: StdMutex<SharedLabels>,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("created_at", &self.created_at)
            .field("user_id", &self.user_id)
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Create a new session with the given ID.
    pub fn new(id: SessionId, user_id: Option<String>) -> Self {
        let now = Utc::now();
        Self {
            id,
            created_at: now,
            last_accessed: RwLock::new(now),
            datasets: Arc::new(RwLock::new(HashMap::new())),
            global_seed: Arc::new(RwLock::new(None)),
            cleaning_sessions: Arc::new(RwLock::new(HashMap::new())),
            spatial_weights: Arc::new(RwLock::new(HashMap::new())),
            memory_profiler: Arc::new(RwLock::new(MemoryProfiler::new())),
            user_id,
            exec_slots: OnceLock::new(),
            active_calls: AtomicUsize::new(0),
            data_bytes: AtomicU64::new(0),
            shared_labels: StdMutex::new(SharedLabels::default()),
        }
    }

    /// Update the last accessed timestamp.
    pub async fn touch(&self) {
        let mut last_accessed = self.last_accessed.write().await;
        *last_accessed = Utc::now();
    }

    /// Check if the session has expired.
    pub async fn is_expired(&self, ttl_minutes: u64) -> bool {
        let last_accessed = *self.last_accessed.read().await;
        is_past_ttl(last_accessed, Utc::now(), ttl_minutes)
    }

    /// Get session info for API responses.
    pub async fn info(&self) -> SessionInfo {
        let datasets = self.datasets.read().await;
        let last_accessed = self.last_accessed.read().await;
        SessionInfo {
            id: self.id.clone(),
            created_at: self.created_at,
            last_accessed: *last_accessed,
            dataset_count: datasets.len(),
            user_id: self.user_id.clone(),
        }
    }

    /// The session's execution semaphore. It is created with `permits`
    /// (clamped to `1..=Semaphore::MAX_PERMITS`) on the first call; later
    /// calls return the same semaphore whatever they pass.
    pub fn exec_semaphore(&self, permits: usize) -> Arc<Semaphore> {
        self.exec_slots
            .get_or_init(|| Arc::new(Semaphore::new(permits.clamp(1, Semaphore::MAX_PERMITS))))
            .clone()
    }

    /// Tool calls currently running in this session.
    pub fn active_calls(&self) -> usize {
        self.active_calls.load(Ordering::SeqCst)
    }

    /// Count a running call until the returned guard is dropped. A session
    /// with a running call is never evicted or expired by the cleanup task.
    pub fn begin_call(self: &Arc<Self>) -> ActiveCallGuard {
        self.active_calls.fetch_add(1, Ordering::SeqCst);
        ActiveCallGuard {
            session: Arc::clone(self),
        }
    }

    /// The cached in-memory size of this session's state (datasets, cleaning
    /// sessions and spatial weights, see [`session_state_bytes`]), in bytes,
    /// as of the last [`refresh_data_bytes`](Self::refresh_data_bytes).
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes.load(Ordering::SeqCst)
    }

    /// Recompute the size of this session's state ([`session_state_bytes`]),
    /// cache it and return it.
    ///
    /// The memory profiler's names are then pruned to the datasets the
    /// session holds (`prune_profiler`). The profiler lives as long as the
    /// session and no quota charges it, so an entry whose dataset left the
    /// session (a replacing `import_session` clears the datasets) must not
    /// stay. When a quota applies, every governed call that changed the
    /// session's state prunes it once the call is kept (`Quotas::enforce`)
    /// or undone ([`restore_state`](Self::restore_state)), or refreshes the
    /// charge after a call that did not finish, so no entry outlives the
    /// call that removed its dataset, however the dataset left.
    pub async fn refresh_data_bytes(&self) -> u64 {
        let bytes = self.recharge_data_bytes().await;
        self.prune_profiler().await;
        bytes
    }

    /// Recompute the size of this session's state ([`session_state_bytes`]),
    /// cache it (its shared labels included) and return it, leaving the
    /// memory profiler alone: for a quota check that may still undo the call
    /// (`Quotas::enforce`), which must not prune the entry of a dataset the
    /// rollback puts back.
    pub(crate) async fn recharge_data_bytes(&self) -> u64 {
        let charge = self.state_charge().await;
        {
            // Both under the mutex, which the server total reads them under,
            // so it never pairs one charge's shared part with another's
            // bytes.
            let mut shared = lock(&self.shared_labels);
            *shared = SharedLabels {
                bytes: charge.shared_labels,
                mappings: charge.mappings,
            };
            self.data_bytes.store(charge.bytes, Ordering::SeqCst);
        }
        charge.bytes
    }

    /// Untrack every memory-profiler entry whose name the session's datasets
    /// lack. The names are collected under the datasets' read lock, which is
    /// dropped before the profiler's lock is taken (`server_stats` holds the
    /// datasets and then reads the profiler). When every tracked name is
    /// held (the common case) nothing else is done, so a refresh that finds
    /// nothing to prune neither lists the profiler's entries nor adds a
    /// history row.
    ///
    /// `load_dataset` tracks its dataset just before inserting it, so a
    /// refresh that runs in between, from another call in the same session,
    /// drops that entry; it is display-only. Under a quota the governor runs
    /// one call per session at a time by default (`ExecConfig::from_config`);
    /// an operator who raises the session concurrency accepts this.
    pub(crate) async fn prune_profiler(&self) {
        let held: HashSet<String> = self.datasets.read().await.keys().cloned().collect();
        let mut profiler = self.memory_profiler.write().await;
        let tracked_and_held = held
            .iter()
            .filter(|name| profiler.dataset_memory(name).is_some())
            .count();
        if tracked_and_held == profiler.dataset_count() {
            return;
        }
        let gone: Vec<String> = profiler
            .stats()
            .datasets
            .into_iter()
            .map(|tracked| tracked.name)
            .filter(|name| !held.contains(name))
            .collect();
        for name in &gone {
            profiler.untrack_dataset(name);
        }
    }

    /// One charge over the session's three maps. Each map is read-locked in
    /// turn and its lock dropped before the next is taken, so this never
    /// holds two of them at once (a handler may hold the datasets and then
    /// take the weights).
    async fn state_charge(&self) -> DataCharge {
        let mut charge = {
            let datasets = self.datasets.read().await;
            datasets_charge(datasets.iter().map(|(name, d)| (name.as_str(), d)))
        };
        self.charge_other_state(&mut charge).await;
        charge
    }

    /// Add the session's cleaning sessions and spatial weights to `charge`,
    /// each map read-locked in turn.
    async fn charge_other_state(&self, charge: &mut DataCharge) {
        charge.add_cleaning_sessions(&*self.cleaning_sessions.read().await);
        charge.add_spatial_weights(&*self.spatial_weights.read().await);
    }

    /// What the session quotas charge for this session's state besides its
    /// datasets: its cleaning sessions and spatial weights, each map
    /// read-locked in turn. A check of a dataset against the quotas adds
    /// this to the datasets' charge; call it without holding the datasets'
    /// lock. A category mapping that a checkpoint shares with a dataset is
    /// then counted on both sides, so the sum errs high.
    pub(crate) async fn other_state_bytes(&self) -> u64 {
        let mut charge = DataCharge::default();
        self.charge_other_state(&mut charge).await;
        charge.bytes
    }

    /// Copy the session's datasets, cleaning sessions and spatial weights,
    /// read-locking each map in turn (never two at once), so that a call can
    /// be undone ([`restore_state`](Self::restore_state)).
    pub async fn snapshot_state(&self) -> SessionStateSnapshot {
        let datasets = self.datasets.read().await.clone();
        let cleaning_sessions = self.cleaning_sessions.read().await.clone();
        let spatial_weights = self.spatial_weights.read().await.clone();
        SessionStateSnapshot {
            datasets,
            cleaning_sessions,
            spatial_weights,
        }
    }

    /// Put `snapshot` back as the session's state, write-locking each map in
    /// turn in the order [`snapshot_state`](Self::snapshot_state) reads
    /// them, then recompute the cached charge (its shared labels included).
    ///
    /// The memory profiler is not part of a snapshot. `load_dataset` tracks
    /// a dataset before inserting it, so a load the quota refuses would
    /// leave its entry behind; the charge refresh at the end prunes the
    /// profiler's names to the restored datasets, as every charge refresh
    /// does ([`refresh_data_bytes`](Self::refresh_data_bytes)). A tracked
    /// name whose dataset the undone call replaced (its columns are not the
    /// snapshot's, [`same_dataset`]) is tracked again with the snapshot's
    /// dataset when its entry holds another estimate, so a load that
    /// replaced a dataset and was undone leaves the original's estimate. A
    /// call that replaced no dataset adds no history row here.
    pub async fn restore_state(&self, snapshot: SessionStateSnapshot) {
        let SessionStateSnapshot {
            datasets,
            cleaning_sessions,
            spatial_weights,
        } = snapshot;
        {
            // The replaced names are collected under the datasets' read
            // lock, which is dropped before the profiler's lock is taken.
            let replaced: Vec<(&String, &Dataset)> = {
                let current = self.datasets.read().await;
                datasets
                    .iter()
                    .filter(|(name, before)| {
                        current
                            .get(name.as_str())
                            .is_some_and(|now| !same_dataset(before, now))
                    })
                    .collect()
            };
            if !replaced.is_empty() {
                let mut profiler = self.memory_profiler.write().await;
                for (name, before) in replaced {
                    let estimate = p2a_core::estimate_dataset_memory(before);
                    if profiler
                        .dataset_memory(name)
                        .is_some_and(|tracked| tracked != estimate)
                    {
                        profiler.track_dataset(name, before);
                    }
                }
            }
        }
        *self.datasets.write().await = datasets;
        *self.cleaning_sessions.write().await = cleaning_sessions;
        *self.spatial_weights.write().await = spatial_weights;
        self.refresh_data_bytes().await;
    }
}

/// Whether `now` is `before` itself, column for column: the same columns,
/// each shared with `before` rather than rebuilt. Only columns backed by a
/// series are compared; any other column (a scalar or partitioned one)
/// counts as changed.
pub(crate) fn same_dataset(before: &Dataset, now: &Dataset) -> bool {
    use p2a_core::polars::prelude::Column;
    let same_column = |a: &Column, b: &Column| match (a, b) {
        (Column::Series(a), Column::Series(b)) => Arc::ptr_eq(&a.0, &b.0),
        _ => false,
    };
    let (before, now) = (before.df().get_columns(), now.df().get_columns());
    before.len() == now.len() && before.iter().zip(now).all(|(a, b)| same_column(a, b))
}

/// A session's datasets, cleaning sessions and spatial weights, copied
/// before a quota-checked call so that it can be undone
/// ([`Session::snapshot_state`], [`Session::restore_state`]).
///
/// The governor takes one before every call when a quota is configured,
/// read-only calls included. It copies the three maps' entries, not the
/// data they hold: each key; each dataset's column handles (the columns
/// are reference-counted), name and source path; each `SpatialWeights` (a
/// clone shares its lists); and each cleaning session's `Arc`. A cleaning
/// tool that then changes a session copies that one session
/// (`Arc::make_mut`), its checkpoints' quality profiles and audit trail
/// included, while the snapshot holds the original.
///
/// The memory profiler behind `server_stats`' history is not part of a
/// snapshot. Its names are pruned to the held datasets at every charge
/// refresh ([`Session::refresh_data_bytes`]), the one
/// [`Session::restore_state`] ends with included, so a restore leaves no
/// entry for a dataset the restored state does not hold.
#[derive(Clone)]
pub struct SessionStateSnapshot {
    pub datasets: HashMap<String, Dataset>,
    pub cleaning_sessions: HashMap<String, Arc<CleaningSession>>,
    pub spatial_weights: HashMap<String, SpatialWeights>,
}

impl fmt::Debug for SessionStateSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionStateSnapshot")
            .field("datasets", &self.datasets.keys().collect::<Vec<_>>())
            .field(
                "cleaning_sessions",
                &self.cleaning_sessions.keys().collect::<Vec<_>>(),
            )
            .field(
                "spatial_weights",
                &self.spatial_weights.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl SessionStateSnapshot {
    /// What the session quotas charge for this state: the datasets
    /// ([`datasets_bytes`]), the cleaning sessions
    /// ([`cleaning_sessions_bytes`]) and the spatial weights
    /// ([`spatial_weights_map_bytes`]), summed (saturating).
    pub fn bytes(&self) -> u64 {
        let mut charge = datasets_charge(self.datasets.iter().map(|(name, d)| (name.as_str(), d)));
        charge.add_cleaning_sessions(&self.cleaning_sessions);
        charge.add_spatial_weights(&self.spatial_weights);
        charge.bytes
    }
}

/// What the session quotas charge for `session`'s state, the figure
/// [`Session::refresh_data_bytes`] caches: its datasets, cleaning sessions
/// and spatial weights, each map read under its lock in turn, without
/// cloning anything.
pub async fn session_state_bytes(session: &Session) -> u64 {
    session.state_charge().await.bytes
}

/// Keeps a session's active-call count raised while it lives.
#[derive(Debug)]
pub struct ActiveCallGuard {
    session: Arc<Session>,
}

impl Drop for ActiveCallGuard {
    fn drop(&mut self) {
        self.session.active_calls.fetch_sub(1, Ordering::SeqCst);
    }
}

// What the dataset quotas charge.
//
// The quotas are meant to bound the memory a session's datasets take, so
// they charge what polars' `estimated_size()` (value buffers only) leaves
// out:
// - the structure every dataset and column costs whatever its length (the
//   `Series`, field, arrow array header and schema entry; the map entry,
//   frame and cached schema), names included; on a 1-row frame it is nearly
//   all of the cost;
// - the 16-byte view of every String and Binary value: polars sizes a view
//   column by its text, so null strings, and short ones a view holds inline,
//   counted nothing;
// - the labels of Categorical and Enum columns, which live in the column's
//   mapping (`estimated_size()` counts the codes). Parquet dictionary<string>
//   columns load as Categorical, all into one process-wide mapping.
// The constants come from RSS growth measured on a release build of the
// hosted server (PR-03, glibc malloc, polars 0.52, 12 cores): per column
// 760 B (1 x 500 generated frames), about 1,100-1,250 B (1 x 12,001 pivots,
// 1 x 12,000 CSV uploads), 2.5 B more per byte of column name; 2.5 KB per
// 1 x 1 dataset; 16.4 B per 8-byte String value and 16.9 B per null one
// (loads of 2,000,000-row files, against the 16 B of a view); and for
// Categorical labels loaded from parquet 1.2-1.4 times their text for
// 1,000-byte labels and 130 B more than their text for 8-byte ones (a
// dictionary column holds its labels twice while it loads, and the allocator
// keeps the freed copy). Each charge is at or above these figures.

/// Bytes charged per dataset on top of its columns.
const DATASET_OVERHEAD_BYTES: u64 = 4096;
/// Bytes charged per column on top of its values.
const COLUMN_OVERHEAD_BYTES: u64 = 1536;
/// Times each byte of a column or dataset name is charged.
const NAME_BYTE_COPIES: u64 = 4;
/// Bytes of one String or Binary view (polars' `View`): the least a value
/// takes, which the check before a file is decoded counts.
pub const VIEW_BYTES: u64 = 16;
/// Bytes charged per String or Binary value: its view and the allocation
/// slack measured around it.
pub(crate) const VIEW_CHARGE_BYTES: u64 = 18;
/// Bytes charged per category label on top of its text: its slots in the
/// mapping (hash table entry, id-to-label entry, allocation rounding).
/// The check before a parquet file is decoded counts it too.
pub(crate) const LABEL_OVERHEAD_BYTES: u64 = 160;

/// What the dataset quotas charge for some datasets (and, for the session
/// quotas, a session's cleaning sessions and spatial weights).
#[derive(Debug, Default)]
struct DataCharge {
    /// The bytes charged.
    bytes: u64,
    /// The part of `bytes` that is labels of Categorical columns, whose
    /// mappings other columns, also other sessions', can share.
    shared_labels: u64,
    /// Those mappings, each once.
    mappings: Vec<Weak<CategoricalMapping>>,
    /// The addresses of `mappings`, so each column's mapping is looked up in
    /// constant time: a parquet column can name a Categories of its own, so
    /// a frame can bring as many mappings as columns. An address stays
    /// unique while the charge lives, since its columns hold the mappings.
    mapping_addresses: HashSet<usize>,
}

impl DataCharge {
    fn add(&mut self, bytes: u64) {
        self.bytes = self.bytes.saturating_add(bytes);
    }

    /// Charge one frame: its structure and each column.
    fn add_frame(&mut self, df: &DataFrame) {
        self.add(structure_bytes(
            df.get_columns().iter().map(|c| c.name().as_str()),
        ));
        for column in df.get_columns() {
            self.add_series(column.as_materialized_series());
        }
    }

    /// Charge one column's values (and, recursively, a nested column's).
    fn add_series(&mut self, s: &Series) {
        match s.dtype() {
            DataType::String => {
                if let Ok(ca) = s.str() {
                    let nulls = ca.null_count() > 0;
                    for arr in ca.downcast_iter() {
                        let bytes =
                            view_bytes(arr.len(), arr.total_buffer_len(), arr.total_bytes_len());
                        self.add(bytes.saturating_add(validity_bytes(arr.len(), nulls)));
                    }
                }
            }
            DataType::Binary => {
                if let Ok(ca) = s.binary() {
                    let nulls = ca.null_count() > 0;
                    for arr in ca.downcast_iter() {
                        let bytes =
                            view_bytes(arr.len(), arr.total_buffer_len(), arr.total_bytes_len());
                        self.add(bytes.saturating_add(validity_bytes(arr.len(), nulls)));
                    }
                }
            }
            DataType::List(_) => {
                if let Ok(ca) = s.list() {
                    for arr in ca.downcast_iter() {
                        let offsets = (arr.len() as u64).saturating_add(1).saturating_mul(8);
                        self.add(
                            offsets.saturating_add(validity_bytes(
                                arr.len(),
                                arr.validity().is_some(),
                            )),
                        );
                    }
                    self.add_series(&ca.get_inner());
                }
            }
            DataType::Array(_, _) => {
                if let Ok(ca) = s.array() {
                    for arr in ca.downcast_iter() {
                        self.add(validity_bytes(arr.len(), arr.validity().is_some()));
                    }
                    self.add_series(&ca.get_inner());
                }
            }
            DataType::Struct(_) => {
                if let Ok(ca) = s.struct_() {
                    for arr in ca.downcast_iter() {
                        self.add(validity_bytes(arr.len(), arr.validity().is_some()));
                    }
                    for field in ca.fields_as_series() {
                        self.add_series(&field);
                    }
                }
            }
            DataType::Categorical(_, mapping) => {
                self.add(s.estimated_size() as u64);
                let labels = used_label_bytes(s, mapping);
                self.add(labels);
                self.shared_labels = self.shared_labels.saturating_add(labels);
                if self.mapping_addresses.insert(mapping_address(mapping)) {
                    self.mappings.push(Arc::downgrade(mapping));
                }
            }
            DataType::Enum(frozen, _) => {
                self.add(s.estimated_size() as u64);
                self.add(frozen_label_bytes(frozen));
            }
            _ => self.add(s.estimated_size() as u64),
        }
    }
}

/// The charge for the structure of a dataset whose columns are named
/// `columns`: the per-dataset and per-column overhead and the column names.
pub fn structure_bytes<'a>(columns: impl IntoIterator<Item = &'a str>) -> u64 {
    columns
        .into_iter()
        .fold(DATASET_OVERHEAD_BYTES, |bytes, name| {
            bytes
                .saturating_add(COLUMN_OVERHEAD_BYTES)
                .saturating_add(name_bytes(name.len()))
        })
}

/// The charge for one column named `name` in a dataset's structure, on top
/// of its values (one of the terms of [`structure_bytes`]).
pub(crate) fn column_structure_bytes(name: &str) -> u64 {
    COLUMN_OVERHEAD_BYTES.saturating_add(name_bytes(name.len()))
}

/// What the dataset quotas charge for the values of the column `s` (see
/// [`datasets_bytes_of`]), its structure left out.
pub(crate) fn series_bytes(s: &Series) -> u64 {
    let mut charge = DataCharge::default();
    charge.add_series(s);
    charge.bytes
}

/// The charge for a name of `len` bytes.
fn name_bytes(len: usize) -> u64 {
    NAME_BYTE_COPIES.saturating_mul(len as u64)
}

/// A validity bitmap of `len` values, when there is one.
fn validity_bytes(len: usize, present: bool) -> u64 {
    if present { len.div_ceil(8) as u64 } else { 0 }
}

/// A view array of `len` values: a view each, and the text kept out of line.
/// That text is the array's buffers, but no more than the text its values
/// hold: a slice or filter shares its source's buffers, which the source is
/// already charged for.
fn view_bytes(len: usize, buffer_len: usize, text_len: usize) -> u64 {
    VIEW_CHARGE_BYTES
        .saturating_mul(len as u64)
        .saturating_add(buffer_len.min(text_len) as u64)
}

/// The labels the codes of the Categorical column `s` use in `mapping`,
/// each once. The mapping can hold far more (all parquet dictionary columns
/// share the global one), so the codes are what decides.
///
/// A column with many values marks its codes in a bitmap of the mapping's
/// size; a column with fewer values than a 64th of the mapping's labels
/// sorts its codes instead, so a short column of a large mapping costs its
/// own length, not the mapping's.
fn used_label_bytes(s: &Series, mapping: &CategoricalMapping) -> u64 {
    let upper = mapping.num_cats_upper_bound();
    let values = s.len() - s.null_count();
    if upper == 0 || values == 0 {
        return 0;
    }
    let label_of = |code: usize| {
        let label = u32::try_from(code)
            .ok()
            .and_then(|code| mapping.cat_to_str(code))
            .map_or(0, str::len);
        label_bytes(label)
    };
    let physical = s.to_physical_repr();
    if values < upper / 64 {
        let mut codes = Vec::with_capacity(values);
        for_each_code(&physical, upper, |code| codes.push(code));
        codes.sort_unstable();
        codes.dedup();
        return codes
            .into_iter()
            .fold(0u64, |bytes, code| bytes.saturating_add(label_of(code)));
    }
    let mut used = vec![0u64; upper.div_ceil(64)];
    for_each_code(&physical, upper, |code| used[code / 64] |= 1 << (code % 64));
    let mut bytes = 0u64;
    for (word_index, &word) in used.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let code = word_index * 64 + word.trailing_zeros() as usize;
            word &= word - 1;
            bytes = bytes.saturating_add(label_of(code));
        }
    }
    bytes
}

/// Call `f` with every code below `upper` of the non-null values of the
/// physical (code) column of a Categorical column (a null slot's code is
/// arbitrary).
fn for_each_code(physical: &Series, upper: usize, f: impl FnMut(usize)) {
    match physical.dtype() {
        DataType::UInt8 => codes_of(physical.u8(), upper, f),
        DataType::UInt16 => codes_of(physical.u16(), upper, f),
        DataType::UInt32 => codes_of(physical.u32(), upper, f),
        _ => {}
    }
}

/// [`for_each_code`] over one code type.
fn codes_of<T>(codes: PolarsResult<&ChunkedArray<T>>, upper: usize, mut f: impl FnMut(usize))
where
    T: PolarsNumericType,
    T::Native: Into<u64>,
{
    let Ok(codes) = codes else {
        return;
    };
    let mut visit = |code: T::Native| {
        if let Ok(code) = usize::try_from(code.into())
            && code < upper
        {
            f(code);
        }
    };
    for arr in codes.downcast_iter() {
        if arr.validity().is_none_or(|v| v.unset_bits() == 0) {
            arr.values().iter().for_each(|&code| visit(code));
        } else {
            arr.iter().flatten().for_each(|&code| visit(code));
        }
    }
}

/// The categories an Enum dtype owns: its label array (views and text) and
/// the mapping's copy of each label. The check before a parquet file is
/// decoded counts it for each Enum leaf column of the file.
pub(crate) fn frozen_label_bytes(frozen: &FrozenCategories) -> u64 {
    let labels = frozen.categories();
    let text = labels.total_bytes_len() as u64;
    view_bytes(
        labels.len(),
        labels.total_buffer_len(),
        labels.total_bytes_len(),
    )
    .saturating_add(label_bytes(0).saturating_mul(labels.len() as u64))
    .saturating_add(text.saturating_add(text / 2))
}

/// The charge for a category label of `len` bytes: its text once and a half
/// (what loading it costs, see above) and its slots in the mapping. The
/// check before a parquet file is decoded counts it for each distinct label
/// of a Categorical column's dictionary pages.
pub(crate) fn label_bytes(len: usize) -> u64 {
    let len = len as u64;
    len.saturating_add(len / 2)
        .saturating_add(LABEL_OVERHEAD_BYTES)
}

/// The address of a mapping, which identifies it while it is held.
fn mapping_address(mapping: &Arc<CategoricalMapping>) -> usize {
    Arc::as_ptr(mapping) as usize
}

/// What a mapping's labels take in memory, counted so far: a mapping only
/// grows (labels get the next code and are never removed), so the next
/// count starts where this one ended.
#[derive(Debug)]
struct LabelCount {
    /// The mapping. Holding it keeps its address from being reused, so the
    /// count found at a live mapping's address is that mapping's.
    mapping: Weak<CategoricalMapping>,
    /// The codes counted: every code below this had its label.
    counted: usize,
    /// What the labels of those codes take.
    bytes: u64,
}

impl LabelCount {
    /// Every label `mapping` holds, used or not: what it takes in memory.
    /// Counts the labels added since the last call. A code whose label is
    /// still being inserted (polars hands out codes before it stores their
    /// labels) is charged as an empty label now and counted again next
    /// time.
    fn update(&mut self, mapping: &CategoricalMapping) -> u64 {
        let upper = mapping.num_cats_upper_bound();
        let mut bytes = self.bytes;
        let mut complete = true;
        for code in self.counted..upper {
            let label = u32::try_from(code)
                .ok()
                .and_then(|code| mapping.cat_to_str(code));
            if label.is_none() && complete {
                complete = false;
                self.counted = code;
                self.bytes = bytes;
            }
            bytes = bytes.saturating_add(label_bytes(label.map_or(0, str::len)));
        }
        if complete {
            self.counted = self.counted.max(upper);
            self.bytes = bytes;
        }
        bytes
    }
}

/// What the dataset quotas charge for a dataset map: [`datasets_bytes_of`]
/// over its entries.
pub fn datasets_bytes(map: &HashMap<String, Dataset>) -> u64 {
    datasets_charge(map.iter().map(|(name, d)| (name.as_str(), d))).bytes
}

/// What the dataset quotas charge for the datasets `entries` (name and
/// dataset): each name, and each frame's values (String and Binary views
/// and Categorical and Enum labels included) and structure (see
/// [`structure_bytes`]).
pub fn datasets_bytes_of<'a>(entries: impl IntoIterator<Item = (&'a str, &'a Dataset)>) -> u64 {
    datasets_charge(entries).bytes
}

/// [`datasets_bytes_of`] with the shared-mapping detail the server total
/// needs.
fn datasets_charge<'a>(entries: impl IntoIterator<Item = (&'a str, &'a Dataset)>) -> DataCharge {
    let mut charge = DataCharge::default();
    for (name, dataset) in entries {
        charge.add(name_bytes(name.len()));
        charge.add_frame(dataset.df());
    }
    charge
}

// What the session quotas charge for a session's other state (MCP-4): the
// cleaning sessions started with `cleaning_session_start` and the spatial
// weights stored by `spatial_neighbors`, which live in the session next to
// its datasets. Each is charged in the same `DataCharge` as the datasets, so
// a category mapping a checkpoint shares with a dataset is counted once in
// the server total. A checkpoint that shares its columns with a dataset (the
// first one always does) is charged for each holder, so the figure is an
// upper bound. Their names and map keys are free-form strings no request
// limit caps, so they are charged one byte per byte.
//
// - Spatial weights: a `SpatialWeights` keeps its own neighbour lists (one
//   index per link) next to its sparse rows (an index and a weight per
//   link), and each observation costs three vector headers. Clones share
//   all of it (T0-WP17), so this is an upper bound. The eigenvalue cache is
//   not charged: the model tools fill it, and charging it would make a model
//   call grow the session and be refused.
// - Cleaning sessions: every checkpoint is a dataset, charged as one (no
//   name), and keeps a quality profile; every cleaning operation that
//   changes a session appends an audit record (failed operations
//   included), with strings the request chose; a refused rollback changes
//   nothing and appends none. The profile's fixed part is charged per
//   column; the top values a profile keeps per string column are copies of
//   that checkpoint's data.

/// Bytes charged per column of every cleaning checkpoint for its quality
/// profile.
pub const CHECKPOINT_PROFILE_BYTES_PER_COLUMN: u64 = 1024;
/// Bytes charged per cleaning audit record, on top of its strings.
pub const AUDIT_RECORD_BYTES: u64 = 256;
/// Bytes charged per neighbour link of spatial weights: its index in the
/// neighbour lists and its index and weight in the sparse rows.
const WEIGHTS_LINK_BYTES: u64 = 24;
/// Bytes charged per observation of spatial weights: the vector headers of
/// its neighbour list and of its sparse row's indices and weights.
const WEIGHTS_ROW_BYTES: u64 = 72;

impl DataCharge {
    /// Charge spatial weights stored under their names.
    fn add_spatial_weights(&mut self, map: &HashMap<String, SpatialWeights>) {
        for (name, weights) in map {
            self.add(name.len() as u64);
            self.add(spatial_weights_bytes(weights));
        }
    }

    /// Charge cleaning sessions stored under their ids.
    fn add_cleaning_sessions(&mut self, map: &HashMap<String, Arc<CleaningSession>>) {
        for (id, session) in map {
            self.add(id.len() as u64);
            self.add_cleaning_session(session);
        }
    }

    /// Charge one cleaning session: its dataset name, each checkpoint (its
    /// frame and its profile), each audit record and its metadata.
    fn add_cleaning_session(&mut self, cs: &CleaningSession) {
        self.add(cs.dataset_name.len() as u64);
        for checkpoint in (0..).map_while(|i| cs.get_checkpoint_dataset(i)) {
            let df = checkpoint.df();
            self.add_frame(df);
            self.add(CHECKPOINT_PROFILE_BYTES_PER_COLUMN.saturating_mul(df.width() as u64));
        }
        for record in &cs.audit_trail {
            self.add(audit_record_bytes(record));
        }
        for (key, value) in &cs.metadata {
            self.add((key.len() as u64).saturating_add(value.len() as u64));
        }
    }
}

/// What the session quotas charge for spatial weights: 24 bytes per
/// neighbour link and 72 per observation (the eigenvalue cache is not
/// charged).
pub fn spatial_weights_bytes(w: &SpatialWeights) -> u64 {
    WEIGHTS_LINK_BYTES
        .saturating_mul(w.neighbors().n_links() as u64)
        .saturating_add(WEIGHTS_ROW_BYTES.saturating_mul(w.n() as u64))
}

/// What the session quotas charge for a session's spatial weights: each
/// name and [`spatial_weights_bytes`].
pub fn spatial_weights_map_bytes(map: &HashMap<String, SpatialWeights>) -> u64 {
    let mut charge = DataCharge::default();
    charge.add_spatial_weights(map);
    charge.bytes
}

/// What the session quotas charge for one cleaning audit record:
/// [`AUDIT_RECORD_BYTES`] and its strings (operation type, description,
/// error, and each parameter's key and value).
pub fn audit_record_bytes(r: &OperationRecord) -> u64 {
    let strings = [
        r.operation_type.len(),
        r.description.len(),
        r.error.as_ref().map_or(0, String::len),
    ]
    .into_iter()
    .chain(r.parameters.iter().flat_map(|(k, v)| [k.len(), v.len()]))
    .fold(0u64, |bytes, len| bytes.saturating_add(len as u64));
    AUDIT_RECORD_BYTES.saturating_add(strings)
}

/// What the session quotas charge for one cleaning session: its dataset
/// name; each checkpoint charged as a dataset is (what
/// `datasets_bytes_of([("", checkpoint)])` gives) plus
/// [`CHECKPOINT_PROFILE_BYTES_PER_COLUMN`] per column; each audit record
/// ([`audit_record_bytes`]); and its metadata strings.
pub fn cleaning_session_bytes(cs: &CleaningSession) -> u64 {
    let mut charge = DataCharge::default();
    charge.add_cleaning_session(cs);
    charge.bytes
}

/// What the session quotas charge for a session's cleaning sessions: each
/// id and [`cleaning_session_bytes`].
pub fn cleaning_sessions_bytes(map: &HashMap<String, Arc<CleaningSession>>) -> u64 {
    let mut charge = DataCharge::default();
    charge.add_cleaning_sessions(map);
    charge.bytes
}

/// The shared-mapping part of a session's cached charge.
#[derive(Debug, Default)]
struct SharedLabels {
    bytes: u64,
    mappings: Vec<Weak<CategoricalMapping>>,
}

fn lock<T>(m: &StdMutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `bytes` as the server prints memory figures, one format for all: the
/// dataset quota refusals (the tool governor's in `exec`, a dataset
/// reload's in `persistent_session`, the check before a file is decoded,
/// [`admit_dataset`], and a munging tool's before it builds), the memory
/// fields of the governor's log lines and `server_stats`. The units are
/// powers of 1024 and are labelled so (KiB, MiB, GiB), since the quotas and
/// the memory soft limit are set in MiB (`P2A_SESSION_DATA_MB` and the
/// others): a refusal names its limit as it was set, "200.00 MiB".
pub(crate) fn fmt_bytes(bytes: u64) -> String {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    // Two decimals: the precision an f64 loses above 2^53 bytes never shows.
    let scaled = |unit: u64| bytes as f64 / unit as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", scaled(GIB))
    } else if bytes >= MIB {
        format!("{:.2} MiB", scaled(MIB))
    } else if bytes >= KIB {
        format!("{:.2} KiB", scaled(KIB))
    } else {
        format!("{bytes} bytes")
    }
}

/// Whether a session last accessed at `last_accessed` has expired at `now`
/// under `ttl_minutes`: more whole minutes have elapsed than the TTL. The
/// comparison is signed, so a `last_accessed` in the future (clock skew, a
/// stored row) is not expired. The live table and the persisted rows use the
/// same rule.
pub(crate) fn is_past_ttl(
    last_accessed: DateTime<Utc>,
    now: DateTime<Utc>,
    ttl_minutes: u64,
) -> bool {
    let elapsed = now.signed_duration_since(last_accessed);
    elapsed.num_minutes() > i64::try_from(ttl_minutes).unwrap_or(i64::MAX)
}

/// The longest `user_id` a session accepts, in bytes.
const MAX_USER_ID_BYTES: usize = 128;

/// Check a client-supplied `user_id`: at most 128 bytes, each an ASCII
/// letter or digit or one of `.`, `_`, `@` and `-`. An empty id is `None`.
pub fn validate_user_id(raw: Option<String>) -> Result<Option<String>, String> {
    let Some(id) = raw.filter(|id| !id.is_empty()) else {
        return Ok(None);
    };
    let allowed = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'@' | b'-');
    if id.len() <= MAX_USER_ID_BYTES && id.bytes().all(allowed) {
        Ok(Some(id))
    } else {
        Err(format!(
            "user_id must be at most {MAX_USER_ID_BYTES} characters from A-Z, a-z, 0-9, '.', \
             '_', '@' and '-'"
        ))
    }
}

/// Whether `id` has the form of a session id: a UUID in the hyphenated
/// lower-case form `SessionManager::create_session` produces.
pub(crate) fn is_session_id(id: &str) -> bool {
    Uuid::parse_str(id).is_ok_and(|u| u.hyphenated().encode_lower(&mut Uuid::encode_buffer()) == id)
}

/// Session information for API responses.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: SessionId,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub dataset_count: usize,
    pub user_id: Option<String>,
}

/// Why a session left the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovalReason {
    /// Deleted by its client.
    Deleted,
    /// Idle longer than the TTL.
    Expired,
    /// Evicted to make room for a new session.
    Evicted,
}

/// Capacity of the removal channel. A receiver that falls further behind
/// sees `RecvError::Lagged` and is expected to resynchronise (the persistent
/// manager runs a sweep).
const REMOVAL_CHANNEL_CAPACITY: usize = 1024;

/// Remove a removed session's data directory, logging a failure.
fn remove_session_dir_logged(id: &str) {
    if let Err(e) = crate::path_jail::remove_session_dir(id) {
        tracing::warn!(session_id = %id, error = %e, "Failed to remove session directory");
    }
}

/// Manages all active sessions.
///
/// This is the only registry of live sessions and the only owner of their
/// `last_accessed` and TTL; the persistent manager holds a shared reference
/// to it and makes stored sessions live again with [`SessionManager::adopt`].
#[derive(Debug)]
pub struct SessionManager {
    /// Active sessions keyed by session ID
    sessions: Arc<RwLock<HashMap<SessionId, Arc<Session>>>>,
    /// Configuration for session management
    config: SessionConfig,
    /// Every removal (delete, expiry, cleanup, eviction) is sent here.
    removals: broadcast::Sender<SessionId>,
    /// Dataset bytes reserved by files being decoded ([`admit_dataset`]).
    reserved_data_bytes: AtomicU64,
    /// The labels counted so far of each shared categorical mapping, by
    /// address, for [`total_data_bytes`](Self::total_data_bytes).
    label_counts: StdMutex<HashMap<usize, LabelCount>>,
}

impl SessionManager {
    /// Create a new session manager with the given configuration.
    pub fn new(config: SessionConfig) -> Self {
        let (removals, _) = broadcast::channel(REMOVAL_CHANNEL_CAPACITY);
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            removals,
            reserved_data_bytes: AtomicU64::new(0),
            label_counts: StdMutex::new(HashMap::new()),
        }
    }

    /// Subscribe to session removals. Every session that leaves the table
    /// (deleted, expired on access or by cleanup, evicted) is sent once,
    /// after it has left. Removals before the call are not replayed.
    pub fn subscribe_removals(&self) -> broadcast::Receiver<SessionId> {
        self.removals.subscribe()
    }

    /// The idle time after which a session expires, in minutes.
    pub fn ttl_minutes(&self) -> u64 {
        self.config.ttl_minutes
    }

    /// A snapshot of every live session's id and last access.
    pub async fn live_last_accessed(&self) -> Vec<(SessionId, DateTime<Utc>)> {
        // Collect first so no last_accessed lock is awaited under the map lock.
        let sessions: Vec<Arc<Session>> = self.sessions.read().await.values().cloned().collect();
        let mut out = Vec::with_capacity(sessions.len());
        for session in sessions {
            let last = *session.last_accessed.read().await;
            out.push((session.id.clone(), last));
        }
        out
    }

    /// Create a new session and return its ID.
    ///
    /// The capacity check and the insert happen under one write lock, so
    /// concurrent creations cannot overshoot `max_sessions`. When the table is
    /// full and `evict_idle_minutes` is set, the least recently used session
    /// that has no running call and has been idle at least that long is
    /// evicted to make room; otherwise this returns `MaxSessionsReached`.
    pub async fn create_session(&self, user_id: Option<String>) -> Result<SessionId, SessionError> {
        let mut sessions = self.sessions.write().await;
        let id = Uuid::new_v4().to_string();
        let session = Arc::new(Session::new(id.clone(), user_id));
        self.admit(&mut sessions, session).await?;

        tracing::info!(session_id = %id, "Created new session");
        Ok(id)
    }

    /// Make a stored session live again under its own id (a persisted session
    /// after a restart). `id` must be a UUID in the hyphenated lower-case form
    /// `create_session` produces; anything else is `NotFound`. A session that
    /// is already live is returned as it is. Otherwise a fresh session
    /// (`last_accessed` = now, no datasets) goes through the same capacity and
    /// eviction rules as `create_session`.
    pub async fn adopt(
        &self,
        id: &str,
        user_id: Option<String>,
    ) -> Result<Arc<Session>, SessionError> {
        if !is_session_id(id) {
            return Err(SessionError::NotFound);
        }
        let mut sessions = self.sessions.write().await;
        if let Some(existing) = sessions.get(id) {
            return Ok(existing.clone());
        }
        let session = Arc::new(Session::new(id.to_string(), user_id));
        self.admit(&mut sessions, session.clone()).await?;
        tracing::info!(session_id = %id, "Adopted stored session");
        Ok(session)
    }

    /// Insert `session` into the write-locked table, evicting the least
    /// recently used idle session first when the table is full and eviction
    /// is on (see [`create_session`](Self::create_session)).
    async fn admit(
        &self,
        sessions: &mut HashMap<SessionId, Arc<Session>>,
        session: Arc<Session>,
    ) -> Result<(), SessionError> {
        if sessions.len() >= self.config.max_sessions {
            // A threshold too large for a chrono::Duration means never evict.
            let Some(threshold) = self
                .config
                .evict_idle_minutes
                .and_then(|m| i64::try_from(m).ok())
                .and_then(chrono::Duration::try_minutes)
            else {
                return Err(SessionError::MaxSessionsReached);
            };
            let now = Utc::now();
            // Lock order: the sessions map, then a session's last_accessed.
            let mut victim: Option<(SessionId, DateTime<Utc>)> = None;
            for (id, session) in sessions.iter() {
                if session.active_calls() > 0 {
                    continue;
                }
                let last = *session.last_accessed.read().await;
                if now.signed_duration_since(last) < threshold {
                    continue;
                }
                if victim.as_ref().is_none_or(|(_, oldest)| last < *oldest) {
                    victim = Some((id.clone(), last));
                }
            }
            let Some((victim_id, _)) = victim else {
                return Err(SessionError::MaxSessionsReached);
            };
            sessions.remove(&victim_id);
            tracing::info!(session_id = %victim_id, "Evicted least recently used idle session");
            self.session_removed(&victim_id, RemovalReason::Evicted);
        }
        sessions.insert(session.id.clone(), session);
        Ok(())
    }

    /// Get a session by ID, updating its last accessed time.
    pub async fn get_session(&self, id: &str) -> Result<Arc<Session>, SessionError> {
        let sessions = self.sessions.read().await;
        let session = sessions.get(id).cloned().ok_or(SessionError::NotFound)?;
        drop(sessions);

        // Check if expired (a session with a running call is not)
        if session.active_calls() == 0 && session.is_expired(self.config.ttl_minutes).await {
            // Remove expired session
            let removed = self.sessions.write().await.remove(id);
            if removed.is_some() {
                tracing::info!(session_id = %id, "Removed expired session");
                self.session_removed(id, RemovalReason::Expired);
            }
            return Err(SessionError::Expired);
        }

        // Update last accessed
        session.touch().await;
        Ok(session)
    }

    /// Delete a session by ID.
    pub async fn delete_session(&self, id: &str) -> Result<(), SessionError> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id).ok_or(SessionError::NotFound)?;
        drop(sessions);
        tracing::info!(session_id = %id, "Deleted session");
        self.session_removed(id, RemovalReason::Deleted);
        Ok(())
    }

    /// List all active sessions.
    pub async fn list_sessions(&self) -> Vec<SessionInfo> {
        // Collect first so no dataset lock is awaited under the map lock.
        let sessions: Vec<Arc<Session>> = self.sessions.read().await.values().cloned().collect();
        let mut infos = Vec::with_capacity(sessions.len());
        for session in sessions {
            infos.push(session.info().await);
        }
        infos
    }

    /// Clean up expired sessions (call periodically). Sessions with a running
    /// call are skipped.
    pub async fn cleanup_expired(&self) -> usize {
        let sessions = self.sessions.read().await;
        let mut expired_ids = Vec::new();

        for (id, session) in sessions.iter() {
            if session.active_calls() == 0 && session.is_expired(self.config.ttl_minutes).await {
                expired_ids.push(id.clone());
            }
        }
        drop(sessions);

        let mut count = 0;
        if !expired_ids.is_empty() {
            let mut sessions = self.sessions.write().await;
            for id in &expired_ids {
                if sessions.remove(id).is_some() {
                    count += 1;
                    self.session_removed(id, RemovalReason::Expired);
                }
            }
            tracing::info!(count = count, "Cleaned up expired sessions");
        }

        count
    }

    /// The single hook every removal path (delete, expiry, cleanup, eviction)
    /// calls once the session has left the table. It tells the removal
    /// subscribers (the persistent manager deletes the session's rows) and
    /// removes the session's data directory: on tokio's blocking pool inside
    /// a runtime, inline outside one. Every async method of the manager only
    /// awaits tokio's runtime-agnostic locks, so any executor can poll it;
    /// `start_cleanup_task`, a plain function that spawns a tokio task, is the
    /// only method that needs a runtime.
    fn session_removed(&self, id: &str, reason: RemovalReason) {
        tracing::debug!(session_id = %id, ?reason, "Session removed");
        // An error only means that nobody is subscribed.
        let _ = self.removals.send(id.to_string());
        let id = id.to_string();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn_blocking(move || remove_session_dir_logged(&id));
            }
            Err(_) => remove_session_dir_logged(&id),
        }
    }

    /// Start background cleanup task.
    pub fn start_cleanup_task(self: Arc<Self>, interval_minutes: u64) {
        let manager = self.clone();
        tokio::spawn(async move {
            let interval = tokio::time::Duration::from_secs(interval_minutes * 60);
            loop {
                tokio::time::sleep(interval).await;
                manager.cleanup_expired().await;
            }
        });
    }

    /// Get session count.
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// The cached state bytes of every session ([`Session::data_bytes`]:
    /// datasets, cleaning sessions, spatial weights), summed, with the labels
    /// of shared categorical mappings counted once each and in full. A label
    /// stays in its mapping while any column of the mapping lives, also after
    /// the dataset that brought it is gone, so the labels a session is
    /// charged for are not all the labels in memory. This holds the map's
    /// read lock only to list the sessions and never takes a session's
    /// dataset lock, so a quota check neither waits behind another session's
    /// long write nor holds up session creation. Its cost grows with the
    /// mappings in use and the labels added since the last call, not with
    /// every label held.
    pub async fn total_data_bytes(&self) -> u64 {
        let sessions: Vec<Arc<Session>> = self.sessions.read().await.values().cloned().collect();
        let mut total = 0u64;
        let mut addresses = HashSet::new();
        let mut mappings: Vec<Arc<CategoricalMapping>> = Vec::new();
        for session in &sessions {
            let shared = lock(&session.shared_labels);
            total = total.saturating_add(session.data_bytes().saturating_sub(shared.bytes));
            for mapping in shared.mappings.iter().filter_map(Weak::upgrade) {
                if addresses.insert(mapping_address(&mapping)) {
                    mappings.push(mapping);
                }
            }
        }
        let mut counts = lock(&self.label_counts);
        // Forget mappings nobody holds any more (their addresses can then be
        // reused).
        counts.retain(|_, count| count.mapping.strong_count() > 0);
        mappings.iter().fold(total, |total, mapping| {
            let count = counts
                .entry(mapping_address(mapping))
                .or_insert_with(|| LabelCount {
                    mapping: Arc::downgrade(mapping),
                    counted: 0,
                    bytes: 0,
                });
            total.saturating_add(count.update(mapping))
        })
    }

    /// Dataset bytes reserved by calls admitting datasets now (files being
    /// decoded, results being built).
    fn reserved_data_bytes(&self) -> u64 {
        self.reserved_data_bytes.load(Ordering::SeqCst)
    }

    /// Reserve `bytes` for a file being decoded; returns everything reserved
    /// now, these bytes included.
    fn reserve_data_bytes(&self, bytes: u64) -> u64 {
        let before = self
            .reserved_data_bytes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |r| {
                Some(r.saturating_add(bytes))
            })
            .unwrap_or(0);
        before.saturating_add(bytes)
    }

    /// Release `bytes` of a reservation.
    fn release_data_bytes(&self, bytes: u64) {
        let _ = self
            .reserved_data_bytes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |r| {
                Some(r.saturating_sub(bytes))
            });
    }
}

/// The handler error code of a dataset-quota refusal made before a dataset
/// was built ([`admit_dataset`]); the governor answers it as
/// `quota_exceeded`, as it does a quota refusal after the call.
///
/// Reserved for those refusals: build them with [`quota_refusal`], and
/// never return this code for anything else, since every handler error
/// carrying it is answered `quota_exceeded`.
pub const DATASET_QUOTA_EXCEEDED: rmcp::model::ErrorCode = rmcp::model::ErrorCode(-32001);

/// A handler's quota refusal: `message` with [`DATASET_QUOTA_EXCEEDED`].
pub fn quota_refusal(message: String) -> rmcp::ErrorData {
    rmcp::ErrorData::new(DATASET_QUOTA_EXCEEDED, message, None)
}

/// The dataset quotas of the governed tool call running on this thread, for
/// checks made before a dataset is built ([`admit_dataset`]). The quotas
/// themselves are checked after the call, on what it built; a file whose
/// decoded size alone passes them would allocate all of it first.
#[derive(Debug)]
pub struct DatasetAdmission {
    per_session: Option<u64>,
    total: Option<u64>,
    session: Arc<Session>,
    sessions: Arc<SessionManager>,
    /// What this call has reserved.
    reserved: AtomicU64,
}

thread_local! {
    static ADMISSION: RefCell<Option<Arc<DatasetAdmission>>> = const { RefCell::new(None) };
}

impl DatasetAdmission {
    /// Install the quotas of a call of `session` on this thread until the
    /// returned guard is dropped (after the call's own quota check).
    #[must_use = "dropping the guard uninstalls the quotas the checks before decoding read"]
    pub fn install(
        per_session: Option<u64>,
        total: Option<u64>,
        session: Arc<Session>,
        sessions: Arc<SessionManager>,
    ) -> AdmissionGuard {
        let admission = Arc::new(Self {
            per_session,
            total,
            session,
            sessions,
            reserved: AtomicU64::new(0),
        });
        let previous = ADMISSION.with(|a| a.replace(Some(Arc::clone(&admission))));
        AdmissionGuard {
            admission,
            previous,
            _thread_bound: PhantomData,
        }
    }
}

/// Keeps a call's [`DatasetAdmission`] installed; dropping it uninstalls it
/// and releases what the call reserved.
///
/// It must be dropped on the thread that installed it, since it restores
/// that thread's admission; it is therefore neither `Send` nor `Sync`, so
/// it cannot be held across an `.await` of a multi-threaded runtime.
#[derive(Debug)]
pub struct AdmissionGuard {
    admission: Arc<DatasetAdmission>,
    previous: Option<Arc<DatasetAdmission>>,
    _thread_bound: PhantomData<*const ()>,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        ADMISSION.with(|a| *a.borrow_mut() = previous);
        let reserved = self.admission.reserved.swap(0, Ordering::SeqCst);
        self.admission.sessions.release_data_bytes(reserved);
    }
}

/// Whether a governed call's dataset quotas are installed on this thread.
pub fn dataset_admission_active() -> bool {
    ADMISSION.with(|a| a.borrow().is_some())
}

/// Admit a dataset that will be stored as `name` and charged at least
/// `at_least` bytes (for a file, what decoding it must allocate), before it
/// is built. It is refused when the session's datasets, with it in place of
/// any dataset of that name, and the session's cleaning sessions and spatial
/// weights would pass the session quota, or when the
/// server's session data and the files other calls are decoding would pass
/// the total. An admitted dataset is reserved against the total until the call
/// ends, so files decoding at once in different sessions cannot together
/// pass it. Without installed quotas (stdio, limits off) it admits
/// everything.
pub async fn admit_dataset(name: &str, at_least: u64) -> Result<(), String> {
    admit_charge(name, at_least).await.map_err(|refused| {
        if refused.per_session {
            format!(
                "Loading this file takes at least {} of memory, so {}. Nothing was loaded; load \
                 a smaller file or start a new session.",
                fmt_bytes(refused.added),
                refused.outcome()
            )
        } else {
            format!(
                "Loading this file takes at least {} of memory, and {}. Nothing was loaded; try \
                 again later or load a smaller file.",
                fmt_bytes(refused.added),
                refused.outcome()
            )
        }
    })
}

/// Why [`admit_charge`] refused a dataset: the quota it would pass, and what
/// the datasets would take with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QuotaShortfall {
    /// What the dataset is charged, its name included.
    pub(crate) added: u64,
    /// What the session's state (for the session quota) or the server's
    /// (for the total) would take with it: datasets, cleaning sessions and
    /// spatial weights.
    pub(crate) reached: u64,
    /// The quota that refused it.
    pub(crate) limit: u64,
    /// Whether that is the session quota, rather than the server total.
    pub(crate) per_session: bool,
}

impl QuotaShortfall {
    /// What the dataset would do, as the quota refusals word it: "this
    /// session's datasets would use X (limit Y per session)", or "the
    /// server's dataset memory would reach X (limit Y)". X is
    /// [`reached`](Self::reached), which includes the session's cleaning
    /// sessions and spatial weights (the admission check adds them), so it
    /// can exceed what the datasets alone take; the texts are kept because
    /// tests in other files assert them.
    pub(crate) fn outcome(&self) -> String {
        if self.per_session {
            format!(
                "this session's datasets would use {} (limit {} per session)",
                fmt_bytes(self.reached),
                fmt_bytes(self.limit)
            )
        } else {
            format!(
                "the server's dataset memory would reach {} (limit {})",
                fmt_bytes(self.reached),
                fmt_bytes(self.limit)
            )
        }
    }
}

/// [`admit_dataset`] with the refusal left to the caller: a tool that is
/// about to build a dataset of `name` charged at most `bytes` words why the
/// dataset is that large itself.
pub(crate) async fn admit_charge(name: &str, bytes: u64) -> Result<(), QuotaShortfall> {
    check_charge(name, bytes, true).await
}

/// Whether [`admit_charge`] would admit a dataset of `name` charged `bytes`
/// now, without reserving anything against the total: for a tool that
/// checks what its result takes at least before work that grows with the
/// result (drawing a sample's rows), and admits the result with
/// [`admit_charge`] once it knows its size.
pub(crate) async fn charge_fits(name: &str, bytes: u64) -> Result<(), QuotaShortfall> {
    check_charge(name, bytes, false).await
}

/// [`admit_charge`] (`reserve`) or [`charge_fits`].
async fn check_charge(name: &str, bytes: u64, reserve: bool) -> Result<(), QuotaShortfall> {
    let Some(admission) = ADMISSION.with(|a| a.borrow().clone()) else {
        return Ok(());
    };
    let added = bytes.saturating_add(name_bytes(name.len()));
    let held = {
        let datasets = admission.session.datasets.read().await;
        datasets_bytes_of(
            datasets
                .iter()
                .filter(|(held, _)| held.as_str() != name)
                .map(|(held, d)| (held.as_str(), d)),
        )
    };
    // The session's cleaning sessions and spatial weights count against its
    // quota too (each map read-locked in turn, the datasets' lock dropped).
    let held = held.saturating_add(admission.session.other_state_bytes().await);
    let after = held.saturating_add(added);
    if let Some(limit) = admission.per_session
        && after > limit
    {
        return Err(QuotaShortfall {
            added,
            reached: after,
            limit,
            per_session: true,
        });
    }
    // The reservation only serves the total, so without one nothing is
    // reserved.
    let Some(limit) = admission.total else {
        return Ok(());
    };
    let sessions = &admission.sessions;
    // Reserve first, so two files decoding at once see each other, and
    // record it in the call's admission at once: its guard then releases it
    // however this future ends, also when it is dropped at the await below
    // or unwinds from it. A check without reserving counts what is reserved
    // so far.
    let others_reserved = if reserve {
        let reserved = sessions.reserve_data_bytes(added);
        admission.reserved.fetch_add(added, Ordering::SeqCst);
        reserved.saturating_sub(added)
    } else {
        sessions.reserved_data_bytes()
    };
    // Read without holding a dataset lock (the total takes the registry's).
    let others = sessions
        .total_data_bytes()
        .await
        .saturating_sub(admission.session.data_bytes());
    let total = others.saturating_add(after).saturating_add(others_reserved);
    if total > limit {
        if reserve {
            admission.reserved.fetch_sub(added, Ordering::SeqCst);
            sessions.release_data_bytes(added);
        }
        return Err(QuotaShortfall {
            added,
            reached: total,
            limit,
            per_session: false,
        });
    }
    Ok(())
}

/// Session management errors.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Session not found")]
    NotFound,
    #[error("Session expired")]
    Expired,
    #[error("Maximum number of sessions reached")]
    MaxSessionsReached,
}

/// A wrapper that provides dataset storage for a session.
/// This is used to make tool handlers work with both stdio (single global store)
/// and HTTP (per-session store) transports.
#[derive(Clone)]
pub struct DatasetStore {
    datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    global_seed: Arc<RwLock<Option<u64>>>,
}

impl DatasetStore {
    /// Create a new dataset store (for stdio transport - single global store).
    pub fn new() -> Self {
        Self {
            datasets: Arc::new(RwLock::new(HashMap::new())),
            global_seed: Arc::new(RwLock::new(None)),
        }
    }

    /// Create from a session (for HTTP transport).
    pub fn from_session(session: &Session) -> Self {
        Self {
            datasets: session.datasets.clone(),
            global_seed: session.global_seed.clone(),
        }
    }

    /// Get the datasets map.
    pub fn datasets(&self) -> &Arc<RwLock<HashMap<String, Dataset>>> {
        &self.datasets
    }

    /// Get the global seed.
    pub fn global_seed(&self) -> &Arc<RwLock<Option<u64>>> {
        &self.global_seed
    }
}

impl Default for DatasetStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple in-memory session store for testing and development.
/// In production, you might want to use Redis or another external store.
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    sessions: RwLock<HashMap<SessionId, Arc<Session>>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn insert(&self, session: Arc<Session>) {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.id.clone(), session);
    }

    pub async fn get(&self, id: &str) -> Option<Arc<Session>> {
        let sessions = self.sessions.read().await;
        sessions.get(id).cloned()
    }

    pub async fn list(&self) -> Vec<Arc<Session>> {
        let sessions = self.sessions.read().await;
        sessions.values().cloned().collect()
    }

    pub async fn remove(&self, id: &str) -> Option<Arc<Session>> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id)
    }
}

// Fixtures of the session-state tests (T2-WP02, MCP-4): spatial weights and
// cleaning sessions kept in a `Session` between tool calls. The values the
// tests pin were computed in R 4.3.3. spdep is not installed, so its
// knearneigh (ties broken by the lower index, as p2a does),
// nb2listw(style = "W") and moran.test(randomisation = TRUE) are written
// out from their definitions; the values are therefore not an R-pinned
// case (no validation case or registry entry). The fixture is the same
// doubles in R and Rust (exact integer arithmetic, then one division and
// one multiplication in the same order):
//
//   n <- 40L; i <- seq_len(n); m <- 2147483647
//   st <- numeric(80); cur <- 20261021
//   for (j in 1:80) { cur <- (16807 * cur) %% m; st[j] <- cur }
//   x <- st[2 * i - 1] / m * 10
//   y <- st[2 * i] / m * 10
//   v <- x + 0.5 * y + ((7L * i) %% 11L) / 10
//   D <- as.matrix(dist(cbind(x, y)))
//   knn <- function(k) {
//     lapply(i, function(a) {
//       d <- D[a, ]; d[a] <- Inf
//       o <- order(d, seq_len(n))          # ties by lower index
//       gap <- d[o[k + 1]] - d[o[k]]
//       stopifnot(gap > 1e-9)              # no tie at the k-th neighbour
//       sort(o[seq_len(k)])
//     })
//   }
//   moran <- function(nb) {
//     W <- matrix(0, n, n)
//     for (a in i) W[a, nb[[a]]] <- 1 / length(nb[[a]])
//     z <- v - mean(v)
//     S0 <- sum(W); S1 <- 0.5 * sum((W + t(W))^2)
//     S2 <- sum((rowSums(W) + colSums(W))^2)
//     I <- (n / S0) * sum(W * outer(z, z)) / sum(z^2)
//     EI <- -1 / (n - 1)
//     b2 <- (sum(z^4) / n) / (sum(z^2) / n)^2
//     VI <- (n * ((n^2 - 3 * n + 3) * S1 - n * S2 + 3 * S0^2) -
//            b2 * ((n^2 - n) * S1 - 2 * n * S2 + 6 * S0^2)) /
//           ((n - 1) * (n - 2) * (n - 3) * S0^2) - EI^2
//     zs <- (I - EI) / sqrt(VI)
//     c(I = I, EI = EI, VI = VI, z = zs, links = sum(lengths(nb)))
//   }
//   cl <- data.frame(id = 1:10,
//                    v = c(1.5, 2.25, NA, 4.75, 5.5, NA, 7.25, 8.5, NA, 10.75))
//   nrow(na.omit(cl))                     # 7
//
// Output (options(digits = 17)), E[I] = -1/39 = -0.02564102564102564:
//   k = 4:  I 0.68715919057620856  Var 0.0095894910636934762
//           z 7.2789719093779635   links 160
//   k = 5:  I 0.65697404814893101  Var 0.0074262267610125639
//           z 7.9212144518526779   links 200
//   k = 20: I 0.29360070720378723  Var 0.00083038505252512487
//           z 11.078472857400241   links 800
//   first row: x 5.7023191897675023, y 8.8786224224039465,
//              v 10.8416304009694748
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::*;
    use polars::prelude::df;

    /// The 40 points of the fixture: a Park-Miller generator from 20261021,
    /// two draws a point, each scaled to [0, 10).
    pub(crate) fn spatial_coords() -> Vec<(f64, f64)> {
        const M: u64 = 2_147_483_647;
        let mut s: u64 = 20_261_021;
        let mut u = Vec::with_capacity(80);
        for _ in 0..80 {
            s = 16_807 * s % M;
            u.push(s as f64 / M as f64 * 10.0);
        }
        (0..40).map(|i| (u[2 * i], u[2 * i + 1])).collect()
    }

    /// The variable the Moran tests use: x + y / 2 plus a noise term.
    pub(crate) fn spatial_v() -> Vec<f64> {
        spatial_coords()
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| x + 0.5 * y + ((7 * (i + 1)) % 11) as f64 / 10.0)
            .collect()
    }

    /// The spatial fixture as a dataset: columns `x`, `y` and `v` (Float64).
    pub(crate) fn spatial_fixture() -> Dataset {
        let coords = spatial_coords();
        let x: Vec<f64> = coords.iter().map(|c| c.0).collect();
        let y: Vec<f64> = coords.iter().map(|c| c.1).collect();
        Dataset::new(df! {"x" => x, "y" => y, "v" => spatial_v()}.unwrap())
    }

    /// Ten rows: `id` 1..=10 (Int64) and `v` (Float64) missing in rows 3, 6
    /// and 9 (1-based); `drop_na` leaves 7 rows, as R's `na.omit` does.
    pub(crate) fn cleaning_fixture() -> Dataset {
        Dataset::new(
            df! {
                "id" => (1..=10i64).collect::<Vec<_>>(),
                "v" => [Some(1.5), Some(2.25), None, Some(4.75), Some(5.5), None,
                        Some(7.25), Some(8.5), None, Some(10.75)],
            }
            .unwrap(),
        )
    }

    /// A fresh session (not in any manager) holding `ds` as `name`.
    pub(crate) async fn session_with(name: &str, ds: Dataset) -> Session {
        let session = Session::new(Uuid::new_v4().to_string(), None);
        session.datasets.write().await.insert(name.to_string(), ds);
        session
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn cfg(max: usize, evict: Option<u64>) -> SessionConfig {
        SessionConfig {
            ttl_minutes: 60,
            max_sessions: max,
            evict_idle_minutes: evict,
        }
    }

    async fn backdate(m: &SessionManager, id: &str, minutes: i64) {
        let s = m.get_session(id).await.unwrap();
        *s.last_accessed.write().await = Utc::now() - chrono::Duration::minutes(minutes);
    }

    async fn futures_count(handles: Vec<tokio::task::JoinHandle<bool>>) -> usize {
        let mut ok = 0;
        for h in handles {
            if h.await.unwrap() {
                ok += 1;
            }
        }
        ok
    }

    #[tokio::test]
    async fn full_manager_evicts_least_recently_used_idle_session() {
        let m = SessionManager::new(cfg(2, Some(10)));
        let a = m.create_session(None).await.unwrap();
        let b = m.create_session(None).await.unwrap();
        backdate(&m, &a, 11).await;
        backdate(&m, &b, 12).await;
        let c = m.create_session(None).await.unwrap();
        assert!(matches!(
            m.get_session(&b).await,
            Err(SessionError::NotFound)
        ));
        assert!(m.get_session(&a).await.is_ok() && m.get_session(&c).await.is_ok());
        assert_eq!(m.session_count().await, 2);
    }

    #[tokio::test]
    async fn full_manager_refuses_when_nobody_is_idle_long_enough() {
        let m = SessionManager::new(cfg(2, Some(10)));
        let a = m.create_session(None).await.unwrap();
        m.create_session(None).await.unwrap();
        backdate(&m, &a, 9).await;
        assert!(matches!(
            m.create_session(None).await,
            Err(SessionError::MaxSessionsReached)
        ));
    }

    #[tokio::test]
    async fn eviction_off_keeps_refusing() {
        let m = SessionManager::new(cfg(1, None));
        let a = m.create_session(None).await.unwrap();
        backdate(&m, &a, 59).await;
        assert!(matches!(
            m.create_session(None).await,
            Err(SessionError::MaxSessionsReached)
        ));
    }

    #[tokio::test]
    async fn huge_eviction_threshold_means_never_evict() {
        // A threshold no chrono::Duration can hold must not panic in
        // create_session; it means "never evict".
        for minutes in [u64::MAX, i64::MAX as u64, (i64::MAX / 60) as u64 + 1] {
            let m = SessionManager::new(cfg(1, Some(minutes)));
            let a = m.create_session(None).await.unwrap();
            backdate(&m, &a, 59).await;
            assert!(
                matches!(
                    m.create_session(None).await,
                    Err(SessionError::MaxSessionsReached)
                ),
                "{minutes}"
            );
        }
    }

    #[test]
    fn exec_semaphore_clamps_its_permits() {
        let s = Session::new("s".into(), None);
        assert_eq!(
            s.exec_semaphore(usize::MAX).available_permits(),
            Semaphore::MAX_PERMITS
        );
        let s = Session::new("s".into(), None);
        assert_eq!(s.exec_semaphore(0).available_permits(), 1);
    }

    #[tokio::test]
    async fn busy_session_is_never_evicted() {
        let m = SessionManager::new(cfg(1, Some(10)));
        let a = m.create_session(None).await.unwrap();
        let s = m.get_session(&a).await.unwrap();
        let _g = s.begin_call();
        backdate(&m, &a, 30).await;
        assert!(matches!(
            m.create_session(None).await,
            Err(SessionError::MaxSessionsReached)
        ));
    }

    #[tokio::test]
    async fn busy_session_survives_expiry_cleanup() {
        let m = SessionManager::new(cfg(10, None));
        let a = m.create_session(None).await.unwrap();
        let s = m.get_session(&a).await.unwrap();
        let g = s.begin_call();
        backdate(&m, &a, 61).await;
        assert_eq!(m.cleanup_expired().await, 0);
        drop(g);
        assert_eq!(s.active_calls(), 0);
        assert_eq!(m.cleanup_expired().await, 1);
        assert!(matches!(
            m.get_session(&a).await,
            Err(SessionError::NotFound)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn capacity_holds_under_concurrent_creation() {
        // invariant; the old read-then-write check could overshoot
        let m = Arc::new(SessionManager::new(cfg(5, None)));
        let hs: Vec<_> = (0..40)
            .map(|_| {
                let m = m.clone();
                tokio::spawn(async move { m.create_session(None).await.is_ok() })
            })
            .collect();
        let ok = futures_count(hs).await;
        assert_eq!(ok, 5);
        assert_eq!(m.session_count().await, 5);
    }

    #[tokio::test]
    async fn adopt_respects_capacity_and_ids() {
        let m = SessionManager::new(cfg(1, None));
        let a = m.create_session(None).await.unwrap();
        // A fixed id with letters a-f, so its uppercase form below differs
        // from it; a random UUID can be all digits, which made that variant
        // adopt the live session.
        let id2 = "6f1c0b9e-3a4d-4c2b-9e7f-0a1b2c3d4e5f".to_string();
        assert!(matches!(
            m.adopt(&id2, None).await,
            Err(SessionError::MaxSessionsReached)
        ));
        m.delete_session(&a).await.unwrap();
        let adopted = m.adopt(&id2, Some("u1".into())).await.unwrap();
        assert_eq!(adopted.user_id.as_deref(), Some("u1"));
        assert!(Arc::ptr_eq(&adopted, &m.get_session(&id2).await.unwrap()));
        // Adopting a live id returns the live session, not a second one.
        assert!(Arc::ptr_eq(&adopted, &m.adopt(&id2, None).await.unwrap()));
        assert_eq!(m.session_count().await, 1);
        for bad in [
            "not-a-uuid".to_string(),
            id2.to_uppercase(),
            id2.replace('-', ""),
            format!("{{{id2}}}"),
        ] {
            assert!(
                matches!(m.adopt(&bad, None).await, Err(SessionError::NotFound)),
                "{bad}"
            );
        }
    }

    #[tokio::test]
    async fn removals_are_broadcast() {
        use tokio::sync::broadcast::error::TryRecvError;
        let m = SessionManager::new(cfg(1, Some(10)));
        let mut rx = m.subscribe_removals();

        let a = m.create_session(None).await.unwrap();
        m.delete_session(&a).await.unwrap();
        assert_eq!(rx.try_recv().unwrap(), a);

        let b = m.create_session(None).await.unwrap();
        backdate(&m, &b, 65).await;
        assert!(matches!(
            m.get_session(&b).await,
            Err(SessionError::Expired)
        ));
        assert_eq!(rx.try_recv().unwrap(), b);

        let c = m.create_session(None).await.unwrap();
        backdate(&m, &c, 65).await;
        assert_eq!(m.cleanup_expired().await, 1);
        assert_eq!(rx.try_recv().unwrap(), c);

        let d = m.create_session(None).await.unwrap();
        backdate(&m, &d, 11).await;
        m.create_session(None).await.unwrap(); // evicts d
        assert_eq!(rx.try_recv().unwrap(), d);

        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn validate_user_id_rules() {
        assert_eq!(validate_user_id(None), Ok(None));
        assert_eq!(validate_user_id(Some(String::new())), Ok(None));
        let max = "a".repeat(128);
        assert_eq!(validate_user_id(Some(max.clone())), Ok(Some(max)));
        assert!(validate_user_id(Some("a".repeat(129))).is_err());
        for ok in ["a.b", "a_b", "a@b", "a-b", "Z9", "alice.o-k_1@example.org"] {
            assert_eq!(
                validate_user_id(Some(ok.into())),
                Ok(Some(ok.into())),
                "{ok}"
            );
        }
        for bad in ["has space", "semi;colon", "ü", "a/b", "a\nb", "a+b", "<x>"] {
            let err = validate_user_id(Some(bad.into())).unwrap_err();
            assert!(err.contains("128"), "{err}");
        }
    }

    #[test]
    fn ttl_comparison_is_signed() {
        let now = Utc::now();
        // A last access in the future (clock skew, a stored row) is not expired.
        assert!(!is_past_ttl(now + chrono::Duration::minutes(90), now, 60));
        assert!(!is_past_ttl(now - chrono::Duration::minutes(60), now, 60));
        assert!(is_past_ttl(now - chrono::Duration::minutes(61), now, 60));
        assert!(!is_past_ttl(
            now - chrono::Duration::days(365),
            now,
            u64::MAX
        ));
    }

    #[tokio::test]
    async fn live_last_accessed_snapshots_every_session() {
        let m = SessionManager::new(cfg(10, None));
        assert_eq!(m.ttl_minutes(), 60);
        let a = m.create_session(None).await.unwrap();
        let b = m.create_session(None).await.unwrap();
        backdate(&m, &a, 7).await;
        let mut snap = m.live_last_accessed().await;
        snap.sort_by_key(|(_, t)| *t);
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].0, a);
        assert_eq!(snap[1].0, b);
        assert!(Utc::now().signed_duration_since(snap[0].1).num_minutes() >= 7);
    }

    #[tokio::test]
    async fn total_data_bytes_sums_cached_session_bytes() {
        let m = SessionManager::new(cfg(10, None));
        let (a, b) = (
            m.create_session(None).await.unwrap(),
            m.create_session(None).await.unwrap(),
        );
        let ds = |n: usize| {
            Dataset::new(df! {"x" => (0..n).map(|i| i as f64 + 0.5).collect::<Vec<_>>()}.unwrap())
        };
        let (sa, sb) = (
            m.get_session(&a).await.unwrap(),
            m.get_session(&b).await.unwrap(),
        );
        sa.datasets.write().await.insert("d".into(), ds(1000));
        sb.datasets.write().await.insert("d".into(), ds(3000));
        let (ba, bb) = (sa.refresh_data_bytes().await, sb.refresh_data_bytes().await);
        assert!(ba >= 8000 && bb >= 24000);
        assert_eq!(sa.data_bytes(), ba);
        assert_eq!(m.total_data_bytes().await, ba + bb);
    }

    /// A dataset map with `datasets` under the names `d0`, `d1`, ...
    fn map_of(datasets: Vec<DataFrame>) -> HashMap<String, Dataset> {
        datasets
            .into_iter()
            .enumerate()
            .map(|(i, df)| (format!("d{i}"), Dataset::new(df)))
            .collect()
    }

    /// One row of `n` float columns whose names are `prefix` plus the index.
    fn one_row_frame(n: usize, prefix: &str) -> DataFrame {
        let columns: Vec<Column> = (0..n)
            .map(|i| Column::new(format!("{prefix}{i}").into(), &[i as f64 + 0.25]))
            .collect();
        DataFrame::new(columns).unwrap()
    }

    // The RSS a 1-row column, a 1-x-1 dataset and a byte of column name
    // were measured to cost on a release build of the hosted server
    // (PR-03): 760 B per column (1 x 500 generated frames), 1,250 B (1 x
    // 12,000 CSV uploads), 2.5 KB per 1-x-1 dataset, 2.54 B per name byte
    // (1,000-byte against 10-byte pivot names).
    const RSS_PER_COLUMN: u64 = 1250;
    const RSS_PER_DATASET: u64 = 2520;
    const RSS_PER_NAME_BYTE: f64 = 2.54;

    #[test]
    fn wide_frames_are_charged_what_their_columns_cost() {
        // A 1 x 2,000 frame: estimated_size() sees 16 kB of values, but each
        // column costs about a kilobyte of structure (Series, field, arrow
        // header, schema entry). Pre-fix the quota charged the 16 kB, so a
        // session could hold hundreds of thousands of columns in its quota.
        let df = one_row_frame(2000, "c");
        let values = df.estimated_size() as u64;
        let charged = datasets_bytes(&map_of(vec![df]));
        assert!(
            charged >= 2000 * RSS_PER_COLUMN,
            "charged {charged}, values {values}"
        );
    }

    #[test]
    fn many_small_datasets_are_charged_what_they_cost() {
        let frames = (0..100).map(|_| one_row_frame(1, "x")).collect();
        let charged = datasets_bytes(&map_of(frames));
        assert!(charged >= 100 * RSS_PER_DATASET, "charged {charged}");
    }

    #[test]
    fn long_column_names_are_charged() {
        // munge_pivot names each output column after a cell value, and a
        // CSV header can carry any names: 200 columns with 1,000-byte names
        // cost about 510 kB more than the same columns named c0..c199.
        let short = datasets_bytes(&map_of(vec![one_row_frame(200, "c")]));
        let long = datasets_bytes(&map_of(vec![one_row_frame(200, &"n".repeat(1000))]));
        let names = 200.0 * 1000.0 * RSS_PER_NAME_BYTE;
        assert!(
            long.saturating_sub(short) as f64 >= names,
            "long {long}, short {short}"
        );
    }

    #[test]
    fn values_still_dominate_a_tall_frame() {
        // The overheads are per column and per dataset, not per value: a
        // 100,000 x 2 frame is charged within a few kilobytes of its values.
        // A regression guard, not evidence of the fix: the old charge
        // (`estimated_size()` alone) passes it too, and it keeps the new
        // overheads from growing with the rows.
        let df = df! {
            "x" => (0..100_000).map(|i| i as f64 * 0.5 + 0.1).collect::<Vec<_>>(),
            "y" => (0..100_000).map(|i| (i % 7) as f64 - 0.3).collect::<Vec<_>>(),
        }
        .unwrap();
        let values = df.estimated_size() as u64;
        let charged = datasets_bytes(&map_of(vec![df]));
        assert!(
            charged >= values && charged - values < 16 * 1024,
            "charged {charged}, values {values}"
        );
    }

    /// What the quotas charge for a dataset of the one column `s`.
    fn charge_of(s: Series) -> u64 {
        datasets_bytes(&map_of(vec![DataFrame::new(vec![s.into()]).unwrap()]))
    }

    #[test]
    fn null_and_short_strings_are_charged_their_views() {
        // polars sizes a String column by its text, but every value also
        // takes a 16-byte view, which holds a short string itself: 100,000
        // null or one-byte strings cost 1.6 MB and were charged 0 or 100 kB.
        let n = 100_000usize;
        let nulls = Series::new("s".into(), vec![None::<&str>; n]);
        let short = Series::new(
            "s".into(),
            (0..n).map(|i| ["a", "b"][i % 2]).collect::<Vec<_>>(),
        );
        for s in [nulls, short] {
            let charged = charge_of(s);
            assert!(charged >= 16 * n as u64, "charged {charged}");
        }
    }

    #[test]
    fn long_strings_are_charged_their_text_and_views_once() {
        let n = 10_000usize;
        let s = Series::new(
            "s".into(),
            (0..n).map(|i| format!("{i:0100}")).collect::<Vec<_>>(),
        );
        let (views, text) = (16 * n as u64, 100 * n as u64);
        let charged = charge_of(s);
        assert!(
            charged >= views + text && charged < views + text + 64 * 1024,
            "charged {charged}"
        );
    }

    #[test]
    fn strings_inside_lists_are_charged_their_views() {
        // 1,000 lists of 50 empty strings: 800 kB of views.
        let lists: Vec<Series> = (0..1000)
            .map(|_| Series::new("".into(), vec![""; 50]))
            .collect();
        let charged = charge_of(Series::new("l".into(), lists));
        assert!(charged >= 16 * 50_000, "charged {charged}");
    }

    /// `labels` distinct labels of `len` bytes as a Categorical column of a
    /// Categories of its own (the tests run in parallel, so none of them
    /// uses the process-wide one).
    fn categorical(labels: usize, len: usize) -> Series {
        let values: Vec<String> = (0..labels).map(|i| format!("{i:0len$}")).collect();
        let dtype =
            DataType::from_categories(Categories::random("".into(), CategoricalPhysical::U32));
        Series::new("k".into(), values).cast(&dtype).unwrap()
    }

    #[test]
    fn categorical_labels_are_charged() {
        // A Categorical column's estimated_size() counts its codes; its
        // labels live in the mapping. A parquet dictionary<string> column
        // loads as Categorical, so pre-fix 2,000 labels of 500 bytes (1 MB)
        // were charged 8 kB while the same text as a String column was
        // charged in full.
        let cat = categorical(2000, 500);
        let string = charge_of(cat.cast(&DataType::String).unwrap());
        let charged = charge_of(cat);
        assert!(charged >= 2000 * 500, "categorical {charged}");
        assert!(
            2 * charged >= string,
            "categorical {charged}, string {string}"
        );
    }

    #[test]
    fn a_categorical_column_is_charged_the_labels_it_uses() {
        // Every column of a Categories shares its mapping (all parquet
        // dictionary columns share the global one), so a column is charged
        // the labels its codes use, not the whole mapping.
        let all = categorical(2000, 500);
        let first = all.slice(0, 1);
        let (full, single) = (charge_of(all), charge_of(first));
        assert!(
            full >= 2000 * 500 && single < 16 * 1024,
            "full {full}, single {single}"
        );
    }

    /// A frame of `n` one-row Categorical columns, each of a Categories of
    /// its own (a parquet column can name one in its field metadata), with
    /// one label of 8 bytes each.
    fn own_mapping_columns(n: usize) -> DataFrame {
        let columns: Vec<Column> = (0..n)
            .map(|i| {
                let dtype = DataType::from_categories(Categories::random(
                    "".into(),
                    CategoricalPhysical::U32,
                ));
                Series::new(format!("c{i}").into(), [format!("{i:08}")])
                    .cast(&dtype)
                    .unwrap()
                    .into()
            })
            .collect();
        DataFrame::new(columns).unwrap()
    }

    #[test]
    fn each_columns_own_mapping_is_charged_once() {
        // Every column brings its own mapping, and a second column of one
        // of them adds nothing to the mapping list. The mappings used to be
        // deduplicated with a scan of the list per column, quadratic in
        // their number; a set of their addresses makes it linear.
        let n = 20_000;
        let mut df = own_mapping_columns(n);
        let again = df.column("c7").unwrap().clone().with_name("again".into());
        df.with_column(again).unwrap();
        let datasets = HashMap::from([("d".to_string(), Dataset::new(df))]);
        let started = std::time::Instant::now();
        let charge = datasets_charge(datasets.iter().map(|(name, d)| (name.as_str(), d)));
        let elapsed = started.elapsed();
        assert_eq!(charge.mappings.len(), n);
        assert_eq!(charge.mapping_addresses.len(), n);
        let per_column = COLUMN_OVERHEAD_BYTES + label_bytes(8);
        assert!(
            charge.bytes >= n as u64 * per_column,
            "charged {}",
            charge.bytes
        );
        assert_eq!(charge.shared_labels, (n as u64 + 1) * label_bytes(8));
        // The scan compared each column's mapping with every one listed
        // before it, 2 x 10^8 comparisons here; the set makes it one lookup
        // per column.
        if !cfg!(debug_assertions) {
            assert!(
                elapsed < std::time::Duration::from_millis(250),
                "{elapsed:?}"
            );
        }
    }

    #[test]
    fn a_short_column_of_a_large_mapping_is_charged_the_labels_it_uses() {
        // A column with fewer values than a 64th of its mapping's labels
        // sorts its codes instead of marking them in a bitmap of the
        // mapping's size; it is charged the same: each label it uses once.
        let all = categorical(100_000, 20);
        let one = all.slice(5, 1);
        assert_eq!(
            used_label_bytes(&one, mapping_of(&one)),
            label_bytes(20),
            "one row"
        );
        // Repeated codes and nulls among a few values.
        let some = Series::new(
            "k".into(),
            [Some(format!("{:020}", 3)), None, Some(format!("{:020}", 3))],
        )
        .cast(all.dtype())
        .unwrap()
        .append(&all.slice(40, 2))
        .unwrap()
        .clone();
        assert_eq!(
            used_label_bytes(&some, mapping_of(&some)),
            3 * label_bytes(20)
        );
        // A long column takes the bitmap and agrees.
        let long = all.slice(0, 10_000);
        assert_eq!(
            used_label_bytes(&long, mapping_of(&long)),
            10_000 * label_bytes(20)
        );
    }

    /// The mapping of the Categorical column `s`.
    fn mapping_of(s: &Series) -> &CategoricalMapping {
        match s.dtype() {
            DataType::Categorical(_, mapping) => mapping,
            other => panic!("not categorical: {other:?}"),
        }
    }

    #[test]
    fn label_counts_resume_where_they_ended() {
        let cats = Categories::random("".into(), CategoricalPhysical::U32);
        let dtype = DataType::from_categories(cats);
        let labels = |range: std::ops::Range<usize>| {
            Series::new(
                "k".into(),
                range.map(|i| format!("{i:010}")).collect::<Vec<_>>(),
            )
            .cast(&dtype)
            .unwrap()
        };
        let first = labels(0..1000);
        let mapping = match first.dtype() {
            DataType::Categorical(_, mapping) => Arc::clone(mapping),
            other => panic!("{other:?}"),
        };
        let mut count = LabelCount {
            mapping: Arc::downgrade(&mapping),
            counted: 0,
            bytes: 0,
        };
        assert_eq!(count.update(&mapping), 1000 * label_bytes(10));
        assert_eq!(count.counted, 1000);
        assert_eq!(count.update(&mapping), 1000 * label_bytes(10));
        let _more = labels(500..1500);
        assert_eq!(count.update(&mapping), 1500 * label_bytes(10));
        assert_eq!(count.counted, 1500);
    }

    #[tokio::test]
    async fn the_total_counts_mappings_once_across_calls_and_forgets_dropped_ones() {
        let m = SessionManager::new(cfg(10, None));
        let s = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let df = own_mapping_columns(300);
        s.datasets
            .write()
            .await
            .insert("d".into(), Dataset::new(df));
        let held = s.refresh_data_bytes().await;
        let total = m.total_data_bytes().await;
        // The session's labels, once in its own charge and once per mapping.
        assert_eq!(total, held);
        assert_eq!(m.total_data_bytes().await, total);
        assert_eq!(lock(&m.label_counts).len(), 300);
        s.datasets.write().await.clear();
        s.refresh_data_bytes().await;
        assert!(m.total_data_bytes().await < 16 * 1024);
        assert!(lock(&m.label_counts).is_empty());
    }

    #[test]
    fn enum_categories_are_charged() {
        // An Enum dtype owns its categories whichever of them the column
        // uses: a 1-row column of an Enum of 1,000 300-byte categories.
        let cats: Vec<String> = (0..1000).map(|i| format!("{i:0300}")).collect();
        let frozen = FrozenCategories::new(cats.iter().map(String::as_str)).unwrap();
        let s = Series::new("e".into(), [cats[0].as_str()])
            .cast(&DataType::from_frozen_categories(frozen))
            .unwrap();
        let charged = charge_of(s);
        assert!(charged >= 1000 * 300, "charged {charged}");
    }

    #[tokio::test]
    async fn total_counts_labels_left_in_a_shared_mapping() {
        // A label stays in its mapping while any column of the mapping
        // lives. A session that loads labels and drops them again leaves
        // them in memory while another session holds a column of the same
        // mapping (every parquet dictionary column uses the global one), so
        // the server total counts each mapping's labels once, whoever holds
        // them.
        let m = SessionManager::new(cfg(10, None));
        let (a, b) = (
            m.create_session(None).await.unwrap(),
            m.create_session(None).await.unwrap(),
        );
        let (sa, sb) = (
            m.get_session(&a).await.unwrap(),
            m.get_session(&b).await.unwrap(),
        );
        let labels = categorical(2000, 500);
        let anchor = labels.slice(0, 1);
        let frame = |s: Series| Dataset::new(DataFrame::new(vec![s.into()]).unwrap());
        sa.datasets.write().await.insert("a".into(), frame(anchor));
        sb.datasets.write().await.insert("l".into(), frame(labels));
        sa.refresh_data_bytes().await;
        sb.refresh_data_bytes().await;
        let both = m.total_data_bytes().await;
        assert!(
            both >= 2000 * 500 && both < sa.data_bytes() + sb.data_bytes(),
            "both {both}: the labels count once"
        );

        sb.datasets.write().await.clear();
        sb.refresh_data_bytes().await;
        let left = m.total_data_bytes().await;
        assert!(
            left >= 2000 * 500,
            "left {left}, session charges {} + {}",
            sa.data_bytes(),
            sb.data_bytes()
        );

        sa.datasets.write().await.clear();
        sa.refresh_data_bytes().await;
        let none = m.total_data_bytes().await;
        assert!(none < 16 * 1024, "none {none}");
    }

    #[tokio::test]
    async fn admission_charges_a_dataset_in_place_of_its_name() {
        let m = Arc::new(SessionManager::new(cfg(10, None)));
        let s = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let df =
            df! {"x" => (0..100_000).map(|i| i as f64 * 0.5 + 0.1).collect::<Vec<_>>()}.unwrap();
        s.datasets
            .write()
            .await
            .insert("x".into(), Dataset::new(df));
        let held = s.refresh_data_bytes().await;
        let limit = held + held / 2;
        let _admission = DatasetAdmission::install(Some(limit), None, s.clone(), m.clone());
        // Loading "x" again replaces it; a second dataset of that size does not fit.
        assert!(admit_dataset("x", held).await.is_ok());
        let refusal = admit_dataset("y", held).await.unwrap_err();
        assert!(refusal.contains("per session"), "{refusal}");
    }

    #[tokio::test]
    async fn admissions_reserve_against_the_total_until_their_call_ends() {
        // Two sessions decoding at once: what one reserved counts against
        // the other's admission until the first call ends.
        let m = Arc::new(SessionManager::new(cfg(10, None)));
        let a = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let b = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let total = Some(10 << 20);
        let six = 6 << 20;
        let first = DatasetAdmission::install(None, total, a.clone(), m.clone());
        admit_dataset("d", six).await.unwrap();
        assert!(m.reserved_data_bytes() >= six);
        {
            let _second = DatasetAdmission::install(None, total, b.clone(), m.clone());
            let refusal = admit_dataset("d", six).await.unwrap_err();
            assert!(refusal.contains("server's dataset memory"), "{refusal}");
        }
        drop(first);
        assert_eq!(m.reserved_data_bytes(), 0);
        let _second = DatasetAdmission::install(None, total, b, m.clone());
        assert!(admit_dataset("d", six).await.is_ok());
    }

    #[tokio::test]
    async fn a_charge_that_fits_is_checked_without_reserving_it() {
        // charge_fits checks as admit_charge does but reserves nothing, so
        // a tool can check part of its result before it admits all of it.
        let m = Arc::new(SessionManager::new(cfg(10, None)));
        let a = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let b = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let total = Some(10 << 20);
        let six = 6 << 20;
        let first = DatasetAdmission::install(None, total, a.clone(), m.clone());
        charge_fits("d", six).await.unwrap();
        charge_fits("d", six).await.unwrap();
        assert_eq!(m.reserved_data_bytes(), 0);
        admit_charge("d", six).await.unwrap();
        let reserved = m.reserved_data_bytes();
        assert!(reserved >= six);
        {
            // Another call's check counts what the first one reserved, and
            // adds nothing to it.
            let _second = DatasetAdmission::install(None, total, b, m.clone());
            let shortfall = charge_fits("d", six).await.unwrap_err();
            assert!(!shortfall.per_session && shortfall.reached > 10 << 20);
            assert_eq!(m.reserved_data_bytes(), reserved);
        }
        drop(first);
        assert_eq!(m.reserved_data_bytes(), 0);
        let _third = DatasetAdmission::install(Some(1 << 20), None, a, m.clone());
        assert!(charge_fits("d", 2 << 20).await.unwrap_err().per_session);
    }

    #[tokio::test]
    async fn an_admission_dropped_while_it_waits_leaves_nothing_reserved() {
        // The admission reserves its bytes against the total before it waits
        // to read the total. Pre-fix the call's guard learned of the
        // reservation only after that wait, so a future dropped there (or
        // unwinding from it) left the bytes reserved for the life of the
        // process, refusing later files.
        let m = Arc::new(SessionManager::new(cfg(10, None)));
        let s = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let guard = DatasetAdmission::install(None, Some(10 << 20), s, m.clone());
        {
            // Holding the registry keeps the total from being read.
            let registry = m.sessions.write().await;
            let mut admit = std::pin::pin!(admit_dataset("d", 1 << 20));
            let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
            assert!(admit.as_mut().poll(&mut cx).is_pending());
            assert!(m.reserved_data_bytes() >= 1 << 20);
            drop(registry);
        }
        drop(guard);
        assert_eq!(m.reserved_data_bytes(), 0);
    }

    #[tokio::test]
    async fn without_a_total_nothing_is_reserved() {
        // The reservation only serves the server total; with a session quota
        // alone an admitted file reserves nothing.
        let m = Arc::new(SessionManager::new(cfg(10, None)));
        let s = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let _admission = DatasetAdmission::install(Some(10 << 20), None, s, m.clone());
        admit_dataset("d", 6 << 20).await.unwrap();
        assert_eq!(m.reserved_data_bytes(), 0);
    }

    #[test]
    fn an_admission_guard_stays_on_its_thread() {
        // The guard restores its thread's admission when dropped, so it must
        // not be `Send`. This only compiles when it is not: for a `Send`
        // type both impls apply and the call is ambiguous.
        trait AmbiguousIfSend<A> {
            fn check() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
        <AdmissionGuard as AmbiguousIfSend<_>>::check();
    }

    #[tokio::test]
    async fn without_installed_quotas_everything_is_admitted() {
        assert!(!dataset_admission_active());
        assert!(admit_dataset("d", u64::MAX).await.is_ok());
    }

    /// Drive `f` on this thread without a Tokio runtime. Uncontended tokio
    /// locks are ready on their first poll, so one poll with a no-op waker
    /// is enough; a future that is not ready then would never be woken, so
    /// the helper panics instead of spinning.
    fn block_on_without_runtime<F: std::future::Future>(f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!(
                "future was not ready on its first poll; it needs a runtime or hit a contended lock"
            ),
        }
    }

    /// Memory figures are powers of 1024 and are labelled so: the quotas
    /// are set in MiB (`P2A_SESSION_DATA_MB`), and a 200 MiB limit printed
    /// as "200.00 MB" named a unit 4.6% smaller than the one counted.
    #[test]
    fn byte_figures_are_labelled_in_binary_units() {
        assert_eq!(fmt_bytes(0), "0 bytes");
        assert_eq!(fmt_bytes(1023), "1023 bytes");
        assert_eq!(fmt_bytes(1536), "1.50 KiB");
        assert_eq!(fmt_bytes(200 << 20), "200.00 MiB");
        assert_eq!(fmt_bytes((1 << 30) - (1 << 20)), "1023.00 MiB");
        assert_eq!(fmt_bytes(3 << 30), "3.00 GiB");
        assert!(fmt_bytes(u64::MAX).ends_with(" GiB"));
    }

    /// The quota refusals name what the datasets would use and the limit in
    /// the unit the limit is set in.
    #[test]
    fn quota_refusals_name_their_figures_in_mib() {
        let mut refused = QuotaShortfall {
            added: 40 << 20,
            reached: 240 << 20,
            limit: 200 << 20,
            per_session: true,
        };
        assert_eq!(
            refused.outcome(),
            "this session's datasets would use 240.00 MiB (limit 200.00 MiB per session)"
        );
        refused.per_session = false;
        refused.limit = 500 << 20;
        refused.reached = 520 << 20;
        assert_eq!(
            refused.outcome(),
            "the server's dataset memory would reach 520.00 MiB (limit 500.00 MiB)"
        );
    }

    /// Row-standardised kNN weights over the 40 fixture points: 40 x k links.
    fn knn_weights(k: usize) -> SpatialWeights {
        use p2a_core::{Neighbors, WeightStyle};
        SpatialWeights::from_neighbors(
            &Neighbors::from_knn(&test_fixtures::spatial_coords(), k),
            WeightStyle::RowStd,
        )
    }

    #[test]
    fn spatial_weights_bytes_counts_links_and_rows() {
        use p2a_core::{Neighbors, WeightStyle};
        // 24 bytes a link (its neighbour index, and its index and weight in
        // the sparse rows) and 72 an observation (three vector headers).
        assert_eq!(spatial_weights_bytes(&knn_weights(4)), 24 * 160 + 72 * 40);
        assert_eq!(spatial_weights_bytes(&knn_weights(4)), 6_720);
        assert_eq!(spatial_weights_bytes(&knn_weights(20)), 22_080);
        let isolates = SpatialWeights::from_neighbors(
            &Neighbors::from_indices(vec![vec![], vec![]]),
            WeightStyle::RowStd,
        );
        assert_eq!(spatial_weights_bytes(&isolates), 144);
        // The name is charged too.
        let map = HashMap::from([("w4".to_string(), knn_weights(4))]);
        assert_eq!(spatial_weights_map_bytes(&map), 6_722);
    }

    #[test]
    fn cleaning_session_bytes_counts_checkpoints_profiles_and_audit() {
        let ds = test_fixtures::cleaning_fixture();
        // A checkpoint is charged as a dataset is (no name), plus 1 KiB per
        // column for its quality profile.
        let frame = datasets_bytes_of([("", &ds)]);
        let one = frame + 1024 * 2;
        let mut cs = CleaningSession::new(ds.clone(), "cl");
        assert_eq!(cleaning_session_bytes(&cs), 2 + one);
        cs.apply_operation("noop", "noop", HashMap::new(), |d| Ok(d.clone()))
            .unwrap();
        // A second checkpoint and an audit record of 256 + 4 + 4 bytes.
        assert_eq!(cleaning_session_bytes(&cs), 2 + 2 * one + 264);
        // A failed operation adds no checkpoint, but its record: 256 + 4 + 4
        // + "boom" 4 + "k" 1 + "vv" 2.
        let failed = cs.apply_operation(
            "fail",
            "fail",
            HashMap::from([("k".to_string(), "vv".to_string())]),
            |_| Err("boom".to_string()),
        );
        assert!(failed.is_err());
        assert_eq!(cleaning_session_bytes(&cs), 2 + 2 * one + 264 + 271);
        let map = HashMap::from([("c1".to_string(), Arc::new(cs.clone()))]);
        assert_eq!(
            cleaning_sessions_bytes(&map),
            2 + cleaning_session_bytes(&cs)
        );
    }

    #[tokio::test]
    async fn snapshot_bytes_match_live_bytes_and_restore_round_trips() {
        let s = test_fixtures::session_with("sp", test_fixtures::spatial_fixture()).await;
        s.spatial_weights
            .write()
            .await
            .insert("w4".into(), knn_weights(4));
        let cs = CleaningSession::new(test_fixtures::cleaning_fixture(), "cl");
        let cs_bytes = cleaning_session_bytes(&cs);
        s.cleaning_sessions
            .write()
            .await
            .insert("c1".into(), Arc::new(cs));
        let snap = s.snapshot_state().await;
        let b = snap.bytes();
        let datasets = datasets_bytes(&*s.datasets.read().await);
        assert_eq!(b, datasets + (2 + cs_bytes) + 6_722);
        assert_eq!(session_state_bytes(&s).await, b);

        s.spatial_weights
            .write()
            .await
            .insert("w20".into(), knn_weights(20));
        s.cleaning_sessions.write().await.remove("c1");
        s.datasets
            .write()
            .await
            .insert("extra".into(), test_fixtures::cleaning_fixture());
        s.restore_state(snap).await;
        let keys = |names: Vec<&String>| {
            let mut names: Vec<String> = names.into_iter().cloned().collect();
            names.sort();
            names
        };
        assert_eq!(keys(s.datasets.read().await.keys().collect()), ["sp"]);
        assert_eq!(
            keys(s.cleaning_sessions.read().await.keys().collect()),
            ["c1"]
        );
        assert_eq!(
            keys(s.spatial_weights.read().await.keys().collect()),
            ["w4"]
        );
        assert_eq!(s.data_bytes(), b);
    }

    /// A snapshot shares each cleaning session instead of copying its
    /// checkpoints' profiles and audit trail. A change made through
    /// `Arc::make_mut`, as the cleaning tools make it, copies the one
    /// session it changes and leaves the snapshot's as it was.
    #[tokio::test]
    async fn a_snapshot_shares_each_cleaning_session() {
        let s = Session::new("share".into(), None);
        s.cleaning_sessions.write().await.insert(
            "c1".into(),
            Arc::new(CleaningSession::new(
                test_fixtures::cleaning_fixture(),
                "cl",
            )),
        );
        let snap = s.snapshot_state().await;
        assert!(Arc::ptr_eq(
            &snap.cleaning_sessions["c1"],
            &s.cleaning_sessions.read().await["c1"]
        ));
        let applied = Arc::make_mut(s.cleaning_sessions.write().await.get_mut("c1").unwrap())
            .apply_operation("noop", "noop", HashMap::new(), |d| Ok(d.clone()));
        assert!(applied.is_ok());
        assert_eq!(snap.cleaning_sessions["c1"].audit_trail.len(), 0);
        assert_eq!(s.cleaning_sessions.read().await["c1"].audit_trail.len(), 1);
        s.restore_state(snap).await;
        assert_eq!(s.cleaning_sessions.read().await["c1"].audit_trail.len(), 0);
    }

    /// The memory profiler is not snapshotted, but a restore leaves it no
    /// entry for a dataset the session does not hold: `load_dataset` tracks
    /// its dataset before inserting it, so a load the quota refuses would
    /// otherwise leave its entry for the life of the session.
    #[tokio::test]
    async fn restore_state_untracks_profiler_entries_the_snapshot_lacks() {
        let s = test_fixtures::session_with("sp", test_fixtures::spatial_fixture()).await;
        let sp = s.datasets.read().await["sp"].clone();
        s.memory_profiler.write().await.track_dataset("sp", &sp);
        let snap = s.snapshot_state().await;
        let count = s.memory_profiler.read().await.dataset_count();
        assert_eq!(count, 1);
        // What a refused load leaves behind before its rollback.
        let ghost = test_fixtures::cleaning_fixture();
        s.memory_profiler
            .write()
            .await
            .track_dataset("ghost", &ghost);
        s.datasets.write().await.insert("ghost".into(), ghost);
        s.restore_state(snap).await;
        let profiler = s.memory_profiler.read().await;
        assert_eq!(profiler.dataset_memory("ghost"), None);
        assert_eq!(profiler.dataset_count(), count);
        assert!(profiler.dataset_memory("sp").is_some());
    }

    /// Every charge refresh prunes the profiler to the held datasets, not
    /// only a rollback's: pre-fix an entry whose dataset left the session
    /// any other way (a replacing import clears the map) stayed for the
    /// life of the session, counted by no quota.
    #[tokio::test]
    async fn refresh_data_bytes_untracks_profiler_entries_of_removed_datasets() {
        let s = test_fixtures::session_with("sp", test_fixtures::spatial_fixture()).await;
        let sp = s.datasets.read().await["sp"].clone();
        let a = test_fixtures::cleaning_fixture();
        {
            let mut profiler = s.memory_profiler.write().await;
            profiler.track_dataset("sp", &sp);
            profiler.track_dataset("a", &a);
        }
        s.datasets.write().await.insert("a".into(), a);
        s.refresh_data_bytes().await;
        // Both held: nothing is pruned, and no history row is added.
        let rows = {
            let profiler = s.memory_profiler.read().await;
            assert!(profiler.dataset_memory("a").is_some());
            assert_eq!(profiler.dataset_count(), 2);
            profiler.history().len()
        };
        s.refresh_data_bytes().await;
        assert_eq!(s.memory_profiler.read().await.history().len(), rows);

        s.datasets.write().await.remove("a");
        s.refresh_data_bytes().await;
        {
            let profiler = s.memory_profiler.read().await;
            assert_eq!(profiler.dataset_memory("a"), None);
            assert!(profiler.dataset_memory("sp").is_some());
            assert_eq!(profiler.dataset_count(), 1);
        }

        // A cleared map, as a replacing import leaves it.
        s.datasets.write().await.clear();
        s.refresh_data_bytes().await;
        assert_eq!(s.memory_profiler.read().await.dataset_count(), 0);
    }

    #[tokio::test]
    async fn refresh_data_bytes_counts_all_session_state() {
        let m = SessionManager::new(cfg(10, None));
        let s = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        s.datasets
            .write()
            .await
            .insert("sp".into(), test_fixtures::spatial_fixture());
        let base = s.refresh_data_bytes().await;
        s.spatial_weights
            .write()
            .await
            .insert("w4".into(), knn_weights(4));
        assert_eq!(s.refresh_data_bytes().await, base + 6_722);
        assert_eq!(s.data_bytes(), base + 6_722);
        assert_eq!(m.total_data_bytes().await, base + 6_722);
    }

    #[tokio::test]
    async fn admission_counts_spatial_weights_and_cleaning_sessions() {
        // The check before a dataset is built counts the session's other
        // state against its quota too: pre-fix it summed the datasets only.
        let m = Arc::new(SessionManager::new(cfg(10, None)));
        let s = m
            .get_session(&m.create_session(None).await.unwrap())
            .await
            .unwrap();
        let df =
            df! {"x" => (0..100_000).map(|i| i as f64 * 0.5 + 0.1).collect::<Vec<_>>()}.unwrap();
        s.datasets
            .write()
            .await
            .insert("x".into(), Dataset::new(df));
        let dx = s.refresh_data_bytes().await;
        let _admission = DatasetAdmission::install(Some(dx + 1_000), None, s.clone(), m.clone());
        // Reloading "x" in place (4 bytes are its name's charge) fits alone...
        assert!(admit_dataset("x", dx - 4).await.is_ok());
        // ...but not next to 6,722 bytes of weights,
        s.spatial_weights
            .write()
            .await
            .insert("w4".into(), knn_weights(4));
        let refusal = admit_dataset("x", dx - 4).await.unwrap_err();
        assert!(refusal.contains("per session"), "{refusal}");
        s.spatial_weights.write().await.clear();
        assert!(admit_dataset("x", dx - 4).await.is_ok());
        // ...nor next to a cleaning session.
        s.cleaning_sessions.write().await.insert(
            "c1".into(),
            Arc::new(CleaningSession::new(
                test_fixtures::cleaning_fixture(),
                "cl",
            )),
        );
        let refusal = admit_dataset("x", dx - 4).await.unwrap_err();
        assert!(refusal.contains("per session"), "{refusal}");
    }

    #[test]
    fn removal_paths_work_outside_a_tokio_runtime() {
        assert!(tokio::runtime::Handle::try_current().is_err());
        let m = SessionManager::new(cfg(1, Some(10)));
        let mut removals = m.subscribe_removals();
        // Deletion.
        let a = block_on_without_runtime(m.create_session(None)).unwrap();
        block_on_without_runtime(m.delete_session(&a)).unwrap();
        assert_eq!(removals.try_recv().unwrap(), a);
        // Eviction when the table is full.
        let b = block_on_without_runtime(m.create_session(None)).unwrap();
        block_on_without_runtime(backdate(&m, &b, 11));
        let c = block_on_without_runtime(m.create_session(None)).unwrap();
        assert_eq!(removals.try_recv().unwrap(), b);
        assert!(block_on_without_runtime(m.get_session(&c)).is_ok());
    }
}
