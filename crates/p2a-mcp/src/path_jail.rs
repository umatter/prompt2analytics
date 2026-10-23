//! Filesystem path jail for tool calls and HTTP file endpoints.
//!
//! All user-supplied paths that touch the filesystem must be run through
//! [`validate_data_path`] (reads) or [`validate_data_path_for_write`] (writes)
//! before being passed to loaders, writers, or database drivers, and writers
//! save through [`write_atomically`]. The data root defaults to the process's
//! current working directory and can be overridden with the `P2A_DATA_ROOT`
//! environment variable, which any exposed deployment should set to a narrow
//! data directory.
//!
//! Semantics:
//! - **Inputs.** A relative input is resolved against the data root (not the
//!   process working directory). An absolute input must canonicalise inside
//!   the root. `..` traversal and intermediate symlinks are resolved, and the
//!   result must lie inside the root.
//! - **Symlinks.** A path whose final component (after component
//!   normalisation, so a trailing `/` cannot make the check follow the link)
//!   is a symbolic link is refused, whether the link is live or dangling. A
//!   data file that is itself a symlink must be named by its real path.
//! - **Writes.** [`validate_data_path_for_write`] also refuses a folder at the
//!   target and a parent folder that does not exist. [`write_atomically`]
//!   writes a `0600` temporary file beside the target and renames it into
//!   place, so a link planted at the target after validation is replaced,
//!   never followed, and a failed write leaves nothing behind.
//! - **Messages.** Errors never contain an absolute server path or an
//!   environment variable name: they echo the caller's input (at most 200
//!   characters) or a [`display_path`] form, which is relative to the root.
//!
//! # Per-session directories
//!
//! With session directories off (local, desktop, stdio use) every path
//! resolves against the single data root. With them on
//! ([`configure_session_dirs`], automatic in the open public mode), the tool
//! governor enters the calling session's scope ([`enter_session_scope`]) on
//! the thread that runs each tool call, and the active root is
//! `<data_root>/sessions/<session_id>/`:
//! - The id must be a UUID. The directory is created on first use with mode
//!   `0700`, and so is its parent `sessions/`, which `--session-dirs on`
//!   reserves. Other sessions' directories and the shared root are outside
//!   the active root and so unreachable.
//! - On the first scoped use in a process, UUID-named entries already under
//!   `sessions/` (left by an earlier process) are removed; other names are
//!   left alone. [`remove_session_dir`] removes a session's directory when
//!   the session goes, and from then on (until 65,536 more sessions have been
//!   removed) the directory is not created again for that id: a call that
//!   was still queued or running when its session went is refused ("needs a
//!   session") instead of leaving a directory behind that nothing would
//!   remove.
//! - Without an active scope every path is refused ("needs a session")
//!   rather than falling back to the shared root. Handlers must therefore not
//!   validate paths from threads or tasks they spawn: those have no scope.
//! - The `session_disk_bytes` budget bounds what a session stores: new files
//!   are refused once the directory has used it, and [`write_atomically`]
//!   refuses a write that would take the directory past it.

use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard, Once, OnceLock};

/// Cached resolved jail root, computed once on first use.
static DATA_ROOT: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Returns the canonical jail root directory.
///
/// Resolution order:
/// 1. `P2A_DATA_ROOT` environment variable (if set and non-empty)
/// 2. The process's current working directory
///
/// The default is deliberately the current working directory, **not** the
/// user's home directory: a public deployment that forgets to set
/// `P2A_DATA_ROOT` should expose only the directory the server was launched
/// from, never the whole of `$HOME`. Operators should set `P2A_DATA_ROOT` to a
/// narrow, dedicated data directory for any exposed deployment.
///
/// The value is cached for the lifetime of the process. This is the global
/// root; with per-session directories on, tool paths resolve inside
/// `<root>/sessions/<session_id>/` instead (see the module documentation).
pub fn allowed_data_root() -> Result<PathBuf, String> {
    DATA_ROOT.get_or_init(resolve_data_root).clone()
}

fn resolve_data_root() -> Result<PathBuf, String> {
    if let Ok(raw) = std::env::var("P2A_DATA_ROOT") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let candidate = PathBuf::from(trimmed);
            return candidate.canonicalize().map_err(|e| {
                format!("P2A_DATA_ROOT={:?} cannot be canonicalized: {}", trimmed, e)
            });
        }
    }
    let cwd = std::env::current_dir()
        .map_err(|e| format!("cannot determine current working directory: {}", e))?;
    cwd.canonicalize()
        .map_err(|e| format!("working directory {:?} cannot be canonicalized: {}", cwd, e))
}

/// The root this thread's paths resolve against: the data root, or the active
/// session's directory when session directories are on.
fn active_root() -> Result<PathBuf, String> {
    active_root_for(
        &data_root()?,
        SESSION_DIRS.load(Ordering::SeqCst),
        current_session_scope().as_deref(),
    )
}

/// Validate a user-supplied path for reading.
///
/// A relative input is resolved against the active root (the data root, or
/// the session's directory), an absolute one must lie inside it, and a
/// final-component symlink is refused (see the module documentation). A path
/// that does not exist yet is accepted when its parent folder exists inside
/// the root.
///
/// Returns the canonical absolute path on success, or a message for the
/// caller (never containing a server path) on rejection.
pub fn validate_data_path(input: &str) -> Result<PathBuf, String> {
    resolve_in(&active_root()?, input)
}

/// Validate a user-supplied path as the target of a write: as
/// [`validate_data_path`], and the target must not be a folder. Inside a
/// session scope with a `session_disk_bytes` budget, a new file is refused
/// once the session's directory has used the quota (overwriting a file passes
/// this check; [`write_atomically`] checks the result).
pub fn validate_data_path_for_write(input: &str) -> Result<PathBuf, String> {
    let root = active_root()?;
    let quota = current_quota(&root).map(|(_, limit)| limit);
    resolve_for_write_in(&root, input, quota)
}

/// Write `target` atomically: `write` fills a new temporary file (mode `0600`)
/// in the target's folder, which is then renamed over the target. A symlink
/// at the target is replaced, never followed, and on every error the
/// temporary file is removed. Inside a session scope with a
/// `session_disk_bytes` budget, the write is refused (and nothing is left
/// behind) when the session's directory would exceed it. `target` should come
/// from [`validate_data_path_for_write`].
pub fn write_atomically(
    target: &Path,
    write: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    let quota = if SESSION_DIRS.load(Ordering::SeqCst) {
        current_quota(&active_root()?)
    } else {
        None
    };
    write_atomically_in(
        target,
        quota.as_ref().map(|(dir, limit)| (dir.as_path(), *limit)),
        write,
    )
}

/// `path` for a message: relative to the active root (`/`-separated, `.` for
/// the root itself), never absolute.
pub fn display_path(path: &Path) -> String {
    match active_root() {
        Ok(root) => display_relative(&root, path),
        Err(_) => display_file_name(path),
    }
}

/// Longest echo of a caller's path in a message, in characters.
const SHOWN_CHARS: usize = 200;

/// The caller's input for a message: at most [`SHOWN_CHARS`] characters, cut
/// at a char boundary, with "…" when cut.
fn shown_input(input: &str) -> String {
    match input.char_indices().nth(SHOWN_CHARS) {
        Some((cut, _)) => format!("{}…", &input[..cut]),
        None => input.to_string(),
    }
}

/// A caller's path for a refusal: a relative input as typed (see
/// [`shown_input`]); an absolute input only by its file name, so a message
/// never carries an absolute server path, not even one the caller supplied.
fn shown_path_input(input: &str) -> String {
    let path = Path::new(input);
    if !path.is_absolute() {
        return shown_input(input);
    }
    match path.file_name() {
        Some(name) => format!("…/{}", shown_input(&name.to_string_lossy())),
        None => "(an absolute path)".to_string(),
    }
}

fn outside_error(input: &str) -> String {
    format!(
        "path `{}` is outside the data directory; use a path relative to it, for example \
         `results.csv`",
        shown_path_input(input)
    )
}

fn unresolvable_error(input: &str) -> String {
    format!(
        "cannot resolve path `{}`; check that its folder exists in the data directory",
        shown_path_input(input)
    )
}

/// Resolve `input` inside `root` (which must be canonical) for reading.
fn resolve_in(root: &Path, input: &str) -> Result<PathBuf, String> {
    if input.is_empty() {
        return Err("path is empty".to_string());
    }
    // Component normalisation drops `.` and a trailing `/`, so the symlink
    // check below looks at the link itself rather than through it.
    let rel: PathBuf = Path::new(input).components().collect();
    let candidate = if rel.is_absolute() {
        rel
    } else {
        root.join(rel)
    };
    let meta = fs::symlink_metadata(&candidate);
    if meta.as_ref().is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(format!(
            "`{}` is a symbolic link; links are not followed",
            shown_path_input(input)
        ));
    }
    let canonical = if meta.is_ok() {
        candidate.canonicalize().map_err(|e| {
            tracing::debug!(path = %candidate.display(), error = %e, "cannot canonicalize");
            unresolvable_error(input)
        })?
    } else {
        // Not there yet (an export target): the parent must exist; the file
        // name is joined to the parent's canonical form. (`file_name` is
        // `None` for a path ending in `..`.)
        let (Some(parent), Some(name)) = (candidate.parent(), candidate.file_name()) else {
            return Err(unresolvable_error(input));
        };
        let parent = parent.canonicalize().map_err(|e| {
            tracing::debug!(path = %parent.display(), error = %e, "cannot canonicalize parent");
            unresolvable_error(input)
        })?;
        parent.join(name)
    };
    if !canonical.starts_with(root) {
        return Err(outside_error(input));
    }
    Ok(canonical)
}

/// [`resolve_in`] for a write target: the target must not be a folder and,
/// with a `quota`, a new file is refused once `root` already uses it up.
fn resolve_for_write_in(root: &Path, input: &str, quota: Option<u64>) -> Result<PathBuf, String> {
    let target = resolve_in(root, input)?;
    let meta = fs::symlink_metadata(&target).ok();
    if meta.as_ref().is_some_and(|m| m.is_dir()) {
        return Err(format!(
            "`{}` is a folder; give a file name, for example `results.csv`",
            shown_path_input(input)
        ));
    }
    if let Some(limit) = quota
        && meta.is_none()
        && dir_usage_bytes(root) >= limit
    {
        return Err(quota_error(&display_relative(root, &target), limit));
    }
    Ok(target)
}

fn quota_error(shown: &str, limit: u64) -> String {
    format!(
        "Cannot save `{shown}`: this session's files would exceed its storage limit of \
         {limit} bytes. Overwrite an existing file or save less data."
    )
}

/// Longest file-name part of a temporary file's name, in bytes, so the whole
/// name stays well inside the usual 255-byte limit.
const TMP_NAME_BYTES: usize = 128;

/// See [`write_atomically`]; with `quota = Some((dir, limit))` the write is
/// also refused (and the temporary file removed) when `dir`'s usage after the
/// rename would exceed `limit`.
fn write_atomically_in(
    target: &Path,
    quota: Option<(&Path, u64)>,
    write: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    let shown = match quota {
        Some((dir, _)) => display_relative(dir, target),
        None => display_file_name(target),
    };
    let cannot_save = || format!("cannot save `{shown}`");
    let (Some(parent), Some(name)) = (target.parent(), target.file_name()) else {
        return Err(cannot_save());
    };
    let name = name.to_string_lossy();
    let mut cut = name.len().min(TMP_NAME_BYTES);
    while !name.is_char_boundary(cut) {
        cut -= 1;
    }
    let tmp = parent.join(format!(
        ".{}.{}.p2a-tmp",
        &name[..cut],
        uuid::Uuid::new_v4()
    ));

    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    if let Err(e) = options.open(&tmp) {
        tracing::debug!(path = %tmp.display(), error = %e, "cannot create a temporary file");
        return Err(cannot_save());
    }

    let remove_tmp = || {
        let _ = fs::remove_file(&tmp);
    };
    if let Err(e) = write(&tmp) {
        remove_tmp();
        return Err(e);
    }
    if let Some((dir, limit)) = quota {
        let replaced = fs::symlink_metadata(target)
            .ok()
            .filter(|m| m.is_file())
            .map_or(0, |m| m.len());
        let projected = dir_usage_bytes(dir).saturating_sub(replaced);
        if projected > limit {
            remove_tmp();
            return Err(quota_error(&shown, limit));
        }
    }
    if let Err(e) = fs::rename(&tmp, target) {
        tracing::debug!(path = %target.display(), error = %e, "cannot rename into place");
        remove_tmp();
        return Err(cannot_save());
    }
    Ok(())
}

/// Most directory entries [`dir_usage_bytes`] visits before it reports the
/// folder as full.
const MAX_USAGE_ENTRIES: usize = 100_000;

/// Bytes of the regular files under `dir`, walked without following links;
/// `u64::MAX` (treated as full) past [`MAX_USAGE_ENTRIES`] entries.
fn dir_usage_bytes(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    let mut entries = 0usize;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(read) = fs::read_dir(&d) else {
            continue;
        };
        for entry in read.flatten() {
            entries += 1;
            if entries > MAX_USAGE_ENTRIES {
                return u64::MAX;
            }
            let Ok(meta) = fs::symlink_metadata(entry.path()) else {
                continue;
            };
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    total
}

/// `path` relative to `root` for a message (`/`-separated, `.` for the root
/// itself); for a path outside the root, only its file name.
fn display_relative(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => {
            let parts: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            shown_input(&parts.join("/"))
        }
        Err(_) => display_file_name(path),
    }
}

/// Only the file name of `path` for a message, or a fixed phrase.
fn display_file_name(path: &Path) -> String {
    match path.file_name() {
        Some(name) => shown_input(&name.to_string_lossy()),
        None => "(outside the data directory)".to_string(),
    }
}

// =============================================================================
// Session scopes and per-session directories
// =============================================================================
//
// The tool governor enters a session scope on the blocking thread that runs
// each HTTP-side tool call. While session directories are on, every path is
// resolved inside that session's directory; without a scope, file access is
// refused (fail closed).

/// The folder under the data root that holds the per-session directories.
const SESSIONS_DIR: &str = "sessions";

/// Whether per-session data directories are enabled (set at startup).
static SESSION_DIRS: AtomicBool = AtomicBool::new(false);

/// Removes stale session directories once, on the first scoped use.
static PURGE: Once = Once::new();

/// How many removed sessions [`REMOVED_SESSIONS`] remembers: the most recent
/// 65,536, a few MiB at most (the hash table can double once under churn).
const MAX_REMOVED_SESSIONS: usize = 1 << 16;
// The module doc, `remove_session_dir`'s doc and the one above state this cap
// as a literal (rustdoc cannot link a private constant from a public item).
const _: () = assert!(
    MAX_REMOVED_SESSIONS == 65_536,
    "update the 65,536 in the path_jail module, remove_session_dir and MAX_REMOVED_SESSIONS docs"
);

/// Session ids (as UUIDs) whose directory has been removed, at most `cap` of
/// them: past the cap, the id removed longest ago is forgotten first. An id
/// is inserted once; inserting it again changes nothing.
#[derive(Debug)]
struct RemovedSessions {
    ids: HashSet<u128>,
    /// The ids in `ids`, removed longest ago first.
    order: VecDeque<u128>,
    cap: usize,
}

impl RemovedSessions {
    /// An empty set that remembers at most `cap` ids (at least one).
    fn new(cap: usize) -> Self {
        Self {
            ids: HashSet::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    fn contains(&self, id: u128) -> bool {
        self.ids.contains(&id)
    }

    /// Remember `id`, first forgetting the oldest id when the set is full (so
    /// the order never grows past the cap). The set is updated before the
    /// order is, so an update stopped part-way can only leave an id
    /// remembered longer, never forget one early.
    fn insert(&mut self, id: u128) {
        if !self.ids.insert(id) {
            return;
        }
        while self.order.len() >= self.cap {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.ids.remove(&oldest);
        }
        self.order.push_back(id);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Sessions whose directory [`remove_session_dir`] has removed, whose
/// directory must therefore not be created again. [`session_dir_in`] checks
/// the set and creates the directory under this lock, and
/// [`remove_session_dir_in`] marks the id under it before removing anything,
/// so a creation either happens before the mark (and is removed) or is
/// refused.
///
/// Session ids are fresh random UUIDs that are never reused, so the set only
/// matters while a call for a removed session is still queued or running.
/// It keeps the last [`MAX_REMOVED_SESSIONS`] removals so that a loop of
/// session creations and deletions cannot grow it without bound. The cost is
/// that a call for a removed session that resolves a path once that many more
/// sessions have been removed can create the directory again, and it then
/// stays (within `session_disk_bytes`) until the next process purges it. The
/// set is a backstop until the governor refuses to run a call whose session
/// is no longer in the `SessionManager` table; once that holds, it is needed
/// only for calls already running when their session goes, and can be pruned
/// by age instead of by count (dropping entries older than `queue_wait` +
/// deadline + `exec::DEADLINE_GRACE`), provided overdue work is also refused
/// file access.
static REMOVED_SESSIONS: LazyLock<Mutex<RemovedSessions>> =
    LazyLock::new(|| Mutex::new(RemovedSessions::new(MAX_REMOVED_SESSIONS)));

/// The removed-session set, also after a panic elsewhere while it was held
/// (see [`RemovedSessions::insert`]: a partial update only remembers an id
/// longer).
fn removed_sessions() -> MutexGuard<'static, RemovedSessions> {
    REMOVED_SESSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

thread_local! {
    /// The session whose scope is active on this thread, if any.
    static SCOPE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Enable or disable per-session data directories. Called at startup from the
/// governor's settings; may be called more than once.
pub fn configure_session_dirs(enabled: bool) {
    SESSION_DIRS.store(enabled, Ordering::SeqCst);
}

/// An active session scope; restores the previous scope on drop, also when
/// unwinding. Not `Send`: it must be dropped on the thread that created it.
#[must_use = "the session scope ends when this guard is dropped"]
#[derive(Debug)]
pub struct SessionScope {
    previous: Option<String>,
    active: bool,
    _not_send: PhantomData<*const ()>,
}

impl Drop for SessionScope {
    fn drop(&mut self) {
        if self.active {
            let previous = self.previous.take();
            // `try_with`: never panic while the thread's locals are torn down.
            let _ = SCOPE.try_with(|s| *s.borrow_mut() = previous);
        }
    }
}

/// Set `session_id` as this thread's scope until the guard is dropped,
/// whether or not session directories are on.
fn scope_guard(session_id: &str) -> SessionScope {
    let previous = SCOPE.with(|s| s.borrow_mut().replace(session_id.to_string()));
    SessionScope {
        previous,
        active: true,
        _not_send: PhantomData,
    }
}

/// Enter `session_id`'s scope on the current thread until the returned guard
/// is dropped: while session directories are on, paths then resolve inside
/// that session's directory. Inert when session directories are off.
pub fn enter_session_scope(session_id: &str) -> SessionScope {
    if SESSION_DIRS.load(Ordering::SeqCst) {
        scope_guard(session_id)
    } else {
        SessionScope {
            previous: None,
            active: false,
            _not_send: PhantomData,
        }
    }
}

/// The session whose scope is active on the current thread, if any.
pub fn current_session_scope() -> Option<String> {
    SCOPE.with(|s| s.borrow().clone())
}

/// Remove a session's data directory. Called (off the async runtime) for every
/// session removal: delete, expiry, cleanup and eviction. `Ok(())` when
/// session directories are off, when the id is not a UUID and when the
/// directory does not exist. Links inside the directory are removed, never
/// followed. From then on the directory is not created again for this id
/// (also when it never existed), so a call that runs after its session went
/// cannot use files; the id is remembered until 65,536 more sessions have
/// been removed.
pub fn remove_session_dir(session_id: &str) -> std::io::Result<()> {
    if !SESSION_DIRS.load(Ordering::SeqCst) {
        return Ok(());
    }
    let data_root = allowed_data_root().map_err(std::io::Error::other)?;
    remove_session_dir_in(&data_root, session_id)
}

/// `id` as a UUID when it is a session id in canonical UUID form (lowercase,
/// hyphenated), so the directory name and the id are the same string.
fn session_uuid(id: &str) -> Option<u128> {
    uuid::Uuid::parse_str(id)
        .ok()
        .filter(|u| {
            u.hyphenated()
                .encode_lower(&mut uuid::Uuid::encode_buffer())
                == id
        })
        .map(|u| u.as_u128())
}

/// The global data root with a caller-safe error; the detail (which names the
/// environment variable and the path) is only logged.
fn data_root() -> Result<PathBuf, String> {
    allowed_data_root().map_err(|e| {
        tracing::warn!(error = %e, "the data root cannot be resolved");
        "the server's data directory is not available".to_string()
    })
}

/// The root paths resolve against: `data_root` when session directories are
/// off, else the scope's session directory; refused when they are on and no
/// scope is active.
fn active_root_for(
    data_root: &Path,
    enabled: bool,
    scope: Option<&str>,
) -> Result<PathBuf, String> {
    if !enabled {
        return Ok(data_root.to_path_buf());
    }
    match scope {
        None => Err(
            "file access needs a session; this server keeps each session's files separate"
                .to_string(),
        ),
        Some(id) => session_dir_in(data_root, id),
    }
}

/// `<data_root>/sessions/<id>`, created on first use with mode `0700` (its
/// parent too). The first call in a process first removes stale session
/// directories left by an earlier process. Refused for a session whose
/// directory [`remove_session_dir_in`] has removed.
fn session_dir_in(data_root: &Path, id: &str) -> Result<PathBuf, String> {
    let Some(uuid) = session_uuid(id) else {
        return Err("this session needs a session id in UUID form to use files".to_string());
    };
    PURGE.call_once(|| {
        purge_stale_session_dirs_in(data_root);
    });
    let sessions = data_root.join(SESSIONS_DIR);
    let dir = sessions.join(id);
    let cannot_prepare = |what: &Path, detail: &dyn std::fmt::Display| {
        tracing::warn!(path = %what.display(), error = %detail, "cannot prepare a session directory");
        "cannot prepare this session's data directory".to_string()
    };
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    {
        // Check and create under the lock the removal marks the id under.
        let removed = removed_sessions();
        if removed.contains(uuid) {
            return Err("this session has ended; file access needs a session".to_string());
        }
        builder.create(&dir).map_err(|e| cannot_prepare(&dir, &e))?;
    }
    for d in [&sessions, &dir] {
        let meta = fs::symlink_metadata(d).map_err(|e| cannot_prepare(d, &e))?;
        if !meta.is_dir() {
            return Err(cannot_prepare(d, &"not a real directory"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o777 != 0o700 {
                fs::set_permissions(d, fs::Permissions::from_mode(0o700))
                    .map_err(|e| cannot_prepare(d, &e))?;
            }
        }
    }
    Ok(dir)
}

/// Remove every UUID-named entry under `<data_root>/sessions/` (directories
/// recursively, without following links); other names are left alone.
/// Returns the number of entries removed.
fn purge_stale_session_dirs_in(data_root: &Path) -> usize {
    let sessions = data_root.join(SESSIONS_DIR);
    if !fs::symlink_metadata(&sessions).is_ok_and(|m| m.is_dir()) {
        return 0;
    }
    let Ok(entries) = fs::read_dir(&sessions) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let is_uuid = entry
            .file_name()
            .to_str()
            .is_some_and(|n| uuid::Uuid::parse_str(n).is_ok());
        if !is_uuid {
            continue;
        }
        let path = entry.path();
        let result = match fs::symlink_metadata(&path) {
            Ok(m) if m.is_dir() => fs::remove_dir_all(&path),
            Ok(_) => fs::remove_file(&path),
            Err(e) => Err(e),
        };
        match result {
            Ok(()) => removed += 1,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "cannot remove a stale session directory")
            }
        }
    }
    if removed > 0 {
        tracing::info!(removed, "removed stale session directories");
    }
    removed
}

/// How many times [`remove_session_dir_in`] tries to remove a directory in
/// which a write in progress created a file while it was being emptied.
const REMOVE_ATTEMPTS: usize = 4;

/// Remove `<data_root>/sessions/<id>` (see [`remove_session_dir`]) and mark
/// the id so that [`session_dir_in`] does not create the directory again.
fn remove_session_dir_in(data_root: &Path, id: &str) -> std::io::Result<()> {
    let Some(uuid) = session_uuid(id) else {
        return Ok(());
    };
    // Mark first, whether or not the directory exists yet: from here on no
    // call can create it, so what is removed below stays removed.
    removed_sessions().insert(uuid);
    let sessions = data_root.join(SESSIONS_DIR);
    if !fs::symlink_metadata(&sessions).is_ok_and(|m| m.is_dir()) {
        return Ok(());
    }
    let dir = sessions.join(id);
    let mut attempt = 1;
    loop {
        let result = match fs::symlink_metadata(&dir) {
            Ok(m) if m.is_dir() => fs::remove_dir_all(&dir),
            Ok(_) => fs::remove_file(&dir),
            Err(e) => Err(e),
        };
        match result {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            // A write that passed its path check before the mark created its
            // temporary file while the folder was being emptied. A write that
            // starts after the mark is refused, so trying again finishes.
            Err(e)
                if e.kind() == std::io::ErrorKind::DirectoryNotEmpty
                    && attempt < REMOVE_ATTEMPTS =>
            {
                attempt += 1;
            }
            other => return other,
        }
    }
}

/// The session storage quota for writes under `root`: `Some` only while a
/// session scope is active and the limits set `session_disk_bytes`.
fn current_quota(root: &Path) -> Option<(PathBuf, u64)> {
    if !SESSION_DIRS.load(Ordering::SeqCst) || current_session_scope().is_none() {
        return None;
    }
    crate::tools::limits::current()
        .session_disk_bytes
        .map(|limit| (root.to_path_buf(), limit))
}

#[cfg(test)]
mod tests {
    //! Every test calls the private functions that take the root explicitly, so
    //! none of them touches `P2A_DATA_ROOT`, the session-dirs switch or the
    //! installed limits, and they stay safe while other lib tests change globals.
    use super::*;
    use std::fs;

    /// A temporary directory and its canonical path.
    fn tmp_root() -> (tempfile::TempDir, PathBuf) {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().canonicalize().unwrap();
        (t, p)
    }

    #[test]
    fn relative_input_resolves_against_root() {
        let (_t, root) = tmp_root();
        fs::write(root.join("sales.csv"), "a\n1\n").unwrap();
        assert_eq!(
            resolve_in(&root, "sales.csv").unwrap(),
            root.join("sales.csv")
        );
        assert_eq!(
            resolve_in(&root, "./sales.csv").unwrap(),
            root.join("sales.csv")
        );
        // not yet existing
        assert_eq!(resolve_in(&root, "new.csv").unwrap(), root.join("new.csv"));
        fs::create_dir(root.join("raw")).unwrap();
        assert_eq!(
            resolve_in(&root, "raw/../sales.csv").unwrap(),
            root.join("sales.csv")
        );
    }

    #[test]
    fn absolute_input_inside_root_is_accepted_outside_refused() {
        let (_t, root) = tmp_root();
        fs::write(root.join("d.csv"), "a\n1\n").unwrap();
        assert_eq!(
            resolve_in(&root, root.join("d.csv").to_str().unwrap()).unwrap(),
            root.join("d.csv")
        );
        let (_o, outside) = tmp_root();
        fs::write(outside.join("x.csv"), "a\n").unwrap();
        for input in [
            outside.join("x.csv").to_str().unwrap().to_string(),
            "../x.csv".into(),
            "/etc/passwd".into(),
        ] {
            let err = resolve_in(&root, &input).unwrap_err();
            assert!(err.contains("is outside the data directory"), "{err}");
            assert!(
                !err.contains(root.to_str().unwrap()),
                "leaks the root: {err}"
            );
        }
        assert!(resolve_in(&root, "").is_err());
        assert!(resolve_in(&root, "missing_dir/x.csv").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn final_symlink_is_refused_dangling_or_live() {
        use std::os::unix::fs::symlink;
        let (_t, root) = tmp_root();
        let (_o, outside) = tmp_root();
        // dangling, points out
        symlink(outside.join("escaped.csv"), root.join("link.csv")).unwrap();
        fs::write(root.join("data.csv"), "a\n1\n").unwrap();
        // live, points in
        symlink(root.join("data.csv"), root.join("alias.csv")).unwrap();
        for input in ["link.csv", "link.csv/", "./link.csv", "alias.csv"] {
            let err = resolve_in(&root, input).unwrap_err();
            assert!(err.contains("is a symbolic link"), "{input}: {err}");
            assert!(resolve_for_write_in(&root, input, None).is_err(), "{input}");
        }
        assert!(!outside.join("escaped.csv").exists());
    }

    #[cfg(unix)]
    #[test]
    fn write_atomically_replaces_a_planted_link_without_following_it() {
        use std::os::unix::fs::symlink;
        let (_t, root) = tmp_root();
        let (_o, outside) = tmp_root();
        let target = resolve_for_write_in(&root, "out.csv", None).unwrap();
        // planted after validation
        symlink(outside.join("victim.csv"), &target).unwrap();
        write_atomically_in(&target, None, |tmp| {
            fs::write(tmp, "x,y\n1,2\n").map_err(|e| e.to_string())
        })
        .unwrap();
        assert!(!outside.join("victim.csv").exists());
        assert!(
            !fs::symlink_metadata(&target)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(&target).unwrap(), "x,y\n1,2\n");
    }

    #[test]
    fn write_atomically_cleans_up_on_error() {
        let (_t, root) = tmp_root();
        let target = root.join("out.csv");
        let err = write_atomically_in(&target, None, |tmp| {
            fs::write(tmp, "partial").unwrap();
            Err("boom".into())
        });
        assert_eq!(err.unwrap_err(), "boom");
        assert_eq!(
            fs::read_dir(&root).unwrap().count(),
            0,
            "temp file left behind"
        );
    }

    #[test]
    fn write_target_that_is_a_directory_is_refused() {
        let (_t, root) = tmp_root();
        fs::create_dir(root.join("dir.csv")).unwrap();
        assert!(resolve_for_write_in(&root, "dir.csv", None).is_err());
    }

    #[test]
    fn display_relative_never_shows_absolute_paths() {
        let (_t, root) = tmp_root();
        assert_eq!(
            display_relative(&root, &root.join("sub").join("a.csv")),
            "sub/a.csv"
        );
        assert_eq!(display_relative(&root, &root), ".");
        let shown = display_relative(&root, Path::new("/etc/passwd"));
        assert!(!shown.starts_with('/'), "{shown}");
        assert!(shown.len() <= 220);
        // char-boundary truncation, no panic
        assert!(shown_input(&"é".repeat(300)).chars().count() <= 201);
    }

    const A: &str = "4b0f5a8e-1c2d-4e3f-9a8b-7c6d5e4f3a2b";

    #[test]
    fn active_root_is_global_when_off_and_session_scoped_when_on() {
        let (_t, data) = tmp_root();
        assert_eq!(active_root_for(&data, false, None).unwrap(), data);
        // switch off: scope ignored
        assert_eq!(active_root_for(&data, false, Some(A)).unwrap(), data);
        let dir = active_root_for(&data, true, Some(A)).unwrap();
        assert_eq!(dir, data.join("sessions").join(A));
        assert!(dir.is_dir());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(data.join("sessions"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
        }
        // fail closed
        let err = active_root_for(&data, true, None).unwrap_err();
        assert!(err.contains("needs a session"), "{err}");
        for bad in ["../etc", "abc", "", "4b0f5a8e/../x"] {
            assert!(active_root_for(&data, true, Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn purge_removes_only_uuid_named_entries() {
        let (_t, data) = tmp_root();
        let stale = data.join("sessions").join(A);
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("old.csv"), "x").unwrap();
        fs::create_dir_all(data.join("sessions").join("notes")).unwrap();
        assert_eq!(purge_stale_session_dirs_in(&data), 1);
        assert!(!stale.exists());
        assert!(data.join("sessions").join("notes").exists());
    }

    /// A session id no other test uses: removing one is recorded process-wide
    /// and would refuse that id's directory in every other test.
    fn fresh_id() -> String {
        uuid::Uuid::new_v4().to_string()
    }

    #[test]
    fn remove_session_dir_in_is_idempotent_and_ignores_non_uuids() {
        let (_t, data) = tmp_root();
        let id = fresh_id();
        let dir = data.join("sessions").join(&id);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.csv"), "x").unwrap();
        remove_session_dir_in(&data, &id).unwrap();
        assert!(!dir.exists());
        // missing: Ok
        remove_session_dir_in(&data, &id).unwrap();
        fs::create_dir_all(data.join("keep")).unwrap();
        // not a UUID: no-op
        remove_session_dir_in(&data, "../keep").unwrap();
        assert!(data.join("keep").exists());
    }

    #[test]
    fn a_removed_session_dir_is_not_recreated_while_remembered() {
        let (_t, data) = tmp_root();
        let id = fresh_id();
        let dir = active_root_for(&data, true, Some(&id)).unwrap();
        fs::write(dir.join("a.csv"), "x\n1\n").unwrap();
        remove_session_dir_in(&data, &id).unwrap();
        // A call that was still queued when the session went resolves its
        // first path after the removal: it must not create the folder again.
        let err = active_root_for(&data, true, Some(&id)).unwrap_err();
        assert!(err.contains("needs a session"), "{err}");
        assert!(
            fs::symlink_metadata(data.join("sessions").join(&id)).is_err(),
            "the removed session's directory was recreated"
        );
        // Also for a session that never used files before it went.
        let unused = fresh_id();
        remove_session_dir_in(&data, &unused).unwrap();
        assert!(active_root_for(&data, true, Some(&unused)).is_err());
        assert!(fs::symlink_metadata(data.join("sessions").join(&unused)).is_err());
        // Other sessions are unaffected.
        assert!(active_root_for(&data, true, Some(&fresh_id())).is_ok());
    }

    /// The removed-session set is bounded: a create/delete loop cannot grow it
    /// past its cap, and it forgets the id removed longest ago first. A local
    /// set, so the global one the other tests rely on is not flushed.
    #[test]
    fn the_removed_session_set_keeps_only_the_latest_removals() {
        let mut removed = RemovedSessions::new(MAX_REMOVED_SESSIONS);
        let cap = MAX_REMOVED_SESSIONS as u128;
        for id in 0..=cap {
            removed.insert(id);
        }
        assert_eq!(removed.len(), MAX_REMOVED_SESSIONS);
        assert_eq!(removed.order.len(), MAX_REMOVED_SESSIONS);
        assert!(!removed.contains(0), "the oldest id is still remembered");
        assert!(removed.contains(1) && removed.contains(cap));

        // An id inserted again neither grows the set nor moves in the order.
        let mut small = RemovedSessions::new(3);
        for id in [10, 11, 12, 10] {
            small.insert(id);
        }
        assert_eq!((small.len(), small.order.len()), (3, 3));
        small.insert(13);
        assert!(!small.contains(10), "a repeated insert moved the id");
        assert!(small.contains(11) && small.contains(12) && small.contains(13));
    }

    #[test]
    fn session_uuid_accepts_only_the_canonical_form() {
        // A fixed id with letters a-f, so its uppercase form differs from it.
        let id = "6f1c0b9e-3a4d-4c2b-9e7f-0a1b2c3d4e5f".to_string();
        assert_eq!(
            session_uuid(&id),
            Some(uuid::Uuid::parse_str(&id).unwrap().as_u128())
        );
        for other in [
            id.to_uppercase(),
            id.replace('-', ""),
            format!("{{{id}}}"),
            format!("urn:uuid:{id}"),
            String::new(),
        ] {
            assert_eq!(session_uuid(&other), None, "{other}");
        }
    }

    #[test]
    fn session_quota_refuses_new_files_and_oversized_writes() {
        let (_t, dir) = tmp_root();
        fs::write(dir.join("a.csv"), vec![b'x'; 3_000]).unwrap();
        let err = resolve_for_write_in(&dir, "new.csv", Some(2_048)).unwrap_err();
        assert!(err.contains("storage limit"), "{err}");
        // overwrite passes the pre-check
        assert!(resolve_for_write_in(&dir, "a.csv", Some(2_048)).is_ok());
        let target = dir.join("big.csv");
        let err = write_atomically_in(&target, Some((&dir, 4_096)), |tmp| {
            fs::write(tmp, vec![b'y'; 5_000]).map_err(|e| e.to_string())
        })
        .unwrap_err();
        assert!(err.contains("storage limit"), "{err}");
        assert!(!target.exists());
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            1,
            "temp file left behind"
        );
        // replacing a.csv (3,000) by 1,000 bytes is allowed even though 3,000 + 1,000 > 3,500
        write_atomically_in(&dir.join("a.csv"), Some((&dir, 3_500)), |tmp| {
            fs::write(tmp, vec![b'z'; 1_000]).map_err(|e| e.to_string())
        })
        .unwrap();
    }

    #[test]
    fn scope_guard_is_thread_local_nested_and_restored() {
        assert_eq!(current_session_scope(), None);
        {
            let _a = scope_guard(A);
            assert_eq!(current_session_scope().as_deref(), Some(A));
            {
                let _b = scope_guard("b");
                assert_eq!(current_session_scope().as_deref(), Some("b"));
            }
            assert_eq!(current_session_scope().as_deref(), Some(A));
            std::thread::spawn(|| assert_eq!(current_session_scope(), None))
                .join()
                .unwrap();
        }
        assert_eq!(current_session_scope(), None);
    }
}
