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
//! It also holds the per-call session-scope interface the tool governor enters
//! on the thread that runs each HTTP-side tool call. For now that is a stub
//! that records the session id; T0-WP09 implements per-session directories.

use std::cell::RefCell;
use std::fs;
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

/// The root every path is resolved against, with a caller-safe error: the
/// detail (which names the environment variable and the path) is only logged.
fn active_root() -> Result<PathBuf, String> {
    allowed_data_root().map_err(|e| {
        tracing::warn!(error = %e, "the data root cannot be resolved");
        "the server's data directory is not available".to_string()
    })
}

/// Validate a user-supplied path for reading.
///
/// A relative input is resolved against the data root, an absolute one must
/// lie inside it, and a final-component symlink is refused (see the module
/// documentation). A path that does not exist yet is accepted when its parent
/// folder exists inside the root.
///
/// Returns the canonical absolute path on success, or a message for the
/// caller (never containing a server path) on rejection.
pub fn validate_data_path(input: &str) -> Result<PathBuf, String> {
    resolve_in(&active_root()?, input)
}

/// Validate a user-supplied path as the target of a write: as
/// [`validate_data_path`], and the target must not be a folder.
pub fn validate_data_path_for_write(input: &str) -> Result<PathBuf, String> {
    resolve_for_write_in(&active_root()?, input, None)
}

/// Write `target` atomically: `write` fills a new temporary file (mode `0600`)
/// in the target's folder, which is then renamed over the target. A symlink
/// at the target is replaced, never followed, and on every error the
/// temporary file is removed. `target` should come from
/// [`validate_data_path_for_write`].
pub fn write_atomically(
    target: &Path,
    write: impl FnOnce(&Path) -> Result<(), String>,
) -> Result<(), String> {
    write_atomically_in(target, None, write)
}

/// `path` for a message: relative to the data root (`/`-separated, `.` for
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

fn outside_error(input: &str) -> String {
    format!(
        "path `{}` is outside the data directory; use a path relative to it, for example \
         `results.csv`",
        shown_input(input)
    )
}

fn unresolvable_error(input: &str) -> String {
    format!(
        "cannot resolve path `{}`; check that its folder exists in the data directory",
        shown_input(input)
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
            shown_input(input)
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
            shown_input(input)
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
}
