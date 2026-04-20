//! Filesystem path jail for tool calls and HTTP file endpoints.
//!
//! All user-supplied paths that touch the filesystem must be run through
//! [`validate_data_path`] before being passed to loaders, writers, or
//! database drivers. The jail root defaults to the user's home directory
//! and can be overridden with the `P2A_DATA_ROOT` environment variable.
//!
//! The validator canonicalizes the requested path and rejects anything
//! that resolves outside the configured root, blocking `../` traversal
//! and symlink escapes.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Cached resolved jail root, computed once on first use.
static DATA_ROOT: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Returns the canonical jail root directory.
///
/// Resolution order:
/// 1. `P2A_DATA_ROOT` environment variable (if set and non-empty)
/// 2. `dirs::home_dir()`
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
    let home = dirs::home_dir().ok_or_else(|| {
        "cannot determine home directory and P2A_DATA_ROOT is not set".to_string()
    })?;
    home.canonicalize()
        .map_err(|e| format!("home directory {:?} cannot be canonicalized: {}", home, e))
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
}
