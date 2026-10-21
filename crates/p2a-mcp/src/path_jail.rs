//! Filesystem path jail for tool calls and HTTP file endpoints.
//!
//! All user-supplied paths that touch the filesystem must be run through
//! [`validate_data_path`] before being passed to loaders, writers, or
//! database drivers. The jail root defaults to the process's current working
//! directory and can be overridden with the `P2A_DATA_ROOT` environment
//! variable, which any exposed deployment should set to a narrow data dir.
//!
//! The validator canonicalizes the requested path and rejects anything
//! that resolves outside the configured root, blocking `../` traversal
//! and symlink escapes.
//!
//! It also holds the per-call session-scope interface the tool governor enters
//! on the thread that runs each HTTP-side tool call. For now that is a stub
//! that records the session id; T0-WP09 implements per-session directories.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

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
/// The value is cached for the lifetime of the process. In test builds the
/// `reset_data_root_for_tests` helper is available.
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

/// Validate a user-supplied path against the jail root.
///
/// Behavior:
/// - Empty input is rejected.
/// - If the path exists, it is canonicalized (resolving `..` and symlinks)
///   and must start with the jail root.
/// - If the path does not yet exist (e.g., export target), the parent
///   directory must exist and canonicalize inside the jail; the filename
///   is appended to the parent's canonical form.
///
/// Returns the canonical absolute path on success, or a human-readable
/// error message on rejection.
pub fn validate_data_path(input: &str) -> Result<PathBuf, String> {
    if input.is_empty() {
        return Err("path is empty".to_string());
    }
    let root = allowed_data_root()?;
    let requested = PathBuf::from(input);
    let canonical = canonicalize_within(&requested)?;
    if !canonical.starts_with(&root) {
        return Err(format!(
            "path {:?} is outside the allowed data root {:?}",
            canonical.display(),
            root.display()
        ));
    }
    Ok(canonical)
}

fn canonicalize_within(requested: &Path) -> Result<PathBuf, String> {
    if requested.exists() {
        return requested
            .canonicalize()
            .map_err(|e| format!("cannot canonicalize {:?}: {}", requested.display(), e));
    }
    let parent = requested.parent().ok_or_else(|| {
        format!(
            "path {:?} has no parent directory component",
            requested.display()
        )
    })?;
    let filename = requested
        .file_name()
        .ok_or_else(|| format!("path {:?} has no filename component", requested.display()))?;
    let parent_canonical = if parent.as_os_str().is_empty() {
        std::env::current_dir().map_err(|e| format!("cannot resolve current directory: {}", e))?
    } else {
        parent.canonicalize().map_err(|e| {
            format!(
                "parent directory {:?} does not exist or cannot be canonicalized: {}",
                parent.display(),
                e
            )
        })?
    };
    Ok(parent_canonical.join(filename))
}

// =============================================================================
// Session scope (stub)
// =============================================================================
//
// The tool governor enters a session scope on the blocking thread that runs
// each HTTP-side tool call, so path validation can later resolve against a
// per-session directory. This is the interface only: it records the session
// id for the current thread when session directories are enabled. T0-WP09
// implements the semantics (per-session roots, write validation, quotas and
// directory removal).

/// Whether per-session data directories are enabled (set at startup).
static SESSION_DIRS_ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The session whose scope is active on this thread, if any.
    static CURRENT_SESSION_SCOPE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Enable or disable per-session data directories. Called at startup from the
/// governor's settings; may be called again (tests restore it).
///
/// Stub: only switches whether [`enter_session_scope`] records the id;
/// T0-WP09 implements the directories.
pub fn configure_session_dirs(enabled: bool) {
    SESSION_DIRS_ENABLED.store(enabled, Ordering::SeqCst);
}

/// An active session scope; restores the previous scope on drop.
#[must_use = "the session scope ends when this guard is dropped"]
#[derive(Debug)]
pub struct SessionScope {
    prev: Option<String>,
    active: bool,
}

impl Drop for SessionScope {
    fn drop(&mut self) {
        if self.active {
            let prev = self.prev.take();
            CURRENT_SESSION_SCOPE.with(|s| *s.borrow_mut() = prev);
        }
    }
}

/// Enter `session_id`'s scope on the current thread until the returned guard
/// is dropped. A no-op when session directories are disabled.
///
/// Stub: records the id only; T0-WP09 makes path validation resolve against
/// the session's directory while a scope is active.
pub fn enter_session_scope(session_id: &str) -> SessionScope {
    if !SESSION_DIRS_ENABLED.load(Ordering::SeqCst) {
        return SessionScope {
            prev: None,
            active: false,
        };
    }
    let prev = CURRENT_SESSION_SCOPE.with(|s| s.borrow_mut().replace(session_id.to_string()));
    SessionScope { prev, active: true }
}

/// The session whose scope is active on the current thread, if any.
pub fn current_session_scope() -> Option<String> {
    CURRENT_SESSION_SCOPE.with(|s| s.borrow().clone())
}

/// Remove a session's data directory. Called (off the async runtime) for every
/// session removal: delete, expiry, cleanup and eviction.
///
/// Stub: returns `Ok(())`; T0-WP09 deletes the directory.
pub fn remove_session_dir(session_id: &str) -> std::io::Result<()> {
    let _ = session_id;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! These tests run as one function to avoid racing on the process-global
    //! `DATA_ROOT` OnceLock. Test-only helpers re-resolve the cached root.
    use super::*;
    use std::fs;

    /// Canonicalize via the unresolved-cache path instead of `allowed_data_root`,
    /// so each assertion sees the P2A_DATA_ROOT currently set in the environment.
    fn validate_with_current_env(input: &str) -> Result<PathBuf, String> {
        if input.is_empty() {
            return Err("path is empty".to_string());
        }
        let root = resolve_data_root()?;
        let requested = PathBuf::from(input);
        let canonical = canonicalize_within(&requested)?;
        if !canonical.starts_with(&root) {
            return Err(format!(
                "path {:?} is outside the allowed data root {:?}",
                canonical.display(),
                root.display()
            ));
        }
        Ok(canonical)
    }

    fn set_env_root(tmp: &Path) {
        // SAFETY: a single-threaded test function owns the environment.
        unsafe {
            std::env::set_var("P2A_DATA_ROOT", tmp);
        }
    }

    #[test]
    fn path_jail_behavior() {
        // 1. Empty input
        assert!(validate_with_current_env("").is_err());

        // 2. File inside root is accepted
        let tmp = tempfile::tempdir().unwrap();
        set_env_root(tmp.path());
        let file = tmp.path().join("data.csv");
        fs::write(&file, "a,b\n1,2\n").unwrap();
        let resolved = validate_with_current_env(file.to_str().unwrap()).unwrap();
        assert!(resolved.starts_with(tmp.path().canonicalize().unwrap()));

        // 3. Traversal escape is rejected
        let escaped = format!("{}/../etc/passwd", tmp.path().display());
        let err = validate_with_current_env(&escaped).unwrap_err();
        assert!(
            err.contains("outside") || err.contains("cannot") || err.contains("canonicalize"),
            "unexpected error message: {}",
            err
        );
    }

    #[test]
    fn session_scope_stub_records_only_when_enabled() {
        configure_session_dirs(false);
        {
            let _s = enter_session_scope("a");
            assert_eq!(current_session_scope(), None);
        }
        configure_session_dirs(true);
        {
            let _a = enter_session_scope("a");
            assert_eq!(current_session_scope().as_deref(), Some("a"));
            {
                let _b = enter_session_scope("b");
                assert_eq!(current_session_scope().as_deref(), Some("b"));
            }
            assert_eq!(current_session_scope().as_deref(), Some("a"));
        }
        assert_eq!(current_session_scope(), None);
        assert!(remove_session_dir("a").is_ok());
        configure_session_dirs(false);
    }
}
