//! Platform-agnostic storage abstraction
//!
//! - Web: Uses localStorage via gloo_storage
//! - Native: Uses file-based storage in the user's config directory. Files are
//!   written through a temp file and a rename; on Unix the file is mode 0600
//!   and the directory 0700, because the settings can hold API keys and the
//!   backend access token.

use serde::{Serialize, de::DeserializeOwned};

/// Write `contents` to `path` so that only the owner can read it.
///
/// The data goes to a fresh temp file in the same directory (created with
/// mode 0600 on Unix), is synced, and is renamed onto `path`. The rename
/// replaces an existing file atomically, which also tightens a file that an
/// older version wrote with default permissions. On failure the temp file is
/// removed. On non-Unix targets only the temp-and-rename part applies; the
/// per-user config directory carries the ACL there.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn write_private_file(path: &std::path::Path, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;

    let file_name = path
        .file_name()
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
        })?
        .to_string_lossy()
        .into_owned();
    let tmp = path.with_file_name(format!("{file_name}.tmp-{}", uuid::Uuid::new_v4()));

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let result = options.open(&tmp).and_then(|mut file| {
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Create `dir` (and its parents) if needed, and make it owner-only (0700)
/// on Unix, tightening a directory that already exists.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn ensure_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Error type for storage operations
#[derive(Debug, Clone)]
pub struct StorageError(pub String);

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StorageError {}

/// Storage backend trait for platform-agnostic persistence
pub trait StorageBackend {
    /// Get a value from storage
    fn get<T: DeserializeOwned>(&self, key: &str) -> Result<T, StorageError>;

    /// Set a value in storage
    fn set<T: Serialize>(&self, key: &str, value: &T) -> Result<(), StorageError>;

    /// Remove a value from storage
    fn remove(&self, key: &str) -> Result<(), StorageError>;

    /// Check if a key exists
    fn contains_key(&self, key: &str) -> bool;
}

// ============================================================================
// Web implementation (WASM with localStorage)
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use gloo_storage::{LocalStorage, Storage};

    /// Web storage backend using localStorage
    pub struct WebStorage;

    impl WebStorage {
        pub fn new() -> Self {
            Self
        }
    }

    impl Default for WebStorage {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StorageBackend for WebStorage {
        fn get<T: DeserializeOwned>(&self, key: &str) -> Result<T, StorageError> {
            LocalStorage::get(key).map_err(|e| StorageError(e.to_string()))
        }

        fn set<T: Serialize>(&self, key: &str, value: &T) -> Result<(), StorageError> {
            LocalStorage::set(key, value).map_err(|e| StorageError(e.to_string()))
        }

        fn remove(&self, key: &str) -> Result<(), StorageError> {
            LocalStorage::delete(key);
            Ok(())
        }

        fn contains_key(&self, key: &str) -> bool {
            LocalStorage::get::<serde_json::Value>(key).is_ok()
        }
    }
}

// ============================================================================
// Native implementation (file-based storage)
// ============================================================================

#[cfg(all(
    not(target_arch = "wasm32"),
    any(feature = "desktop", feature = "mobile")
))]
mod native {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Native storage backend using file system
    pub struct NativeStorage {
        storage_dir: PathBuf,
    }

    impl NativeStorage {
        pub fn new() -> Self {
            let storage_dir = dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("p2a");

            // Ensure the directory exists and is owner-only
            if let Err(e) = ensure_private_dir(&storage_dir) {
                tracing::warn!("Failed to create storage directory: {}", e);
            }

            Self { storage_dir }
        }

        fn key_path(&self, key: &str) -> PathBuf {
            // Sanitize key to be a valid filename
            let safe_key = key.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
            self.storage_dir.join(format!("{}.json", safe_key))
        }
    }

    impl Default for NativeStorage {
        fn default() -> Self {
            Self::new()
        }
    }

    impl StorageBackend for NativeStorage {
        fn get<T: DeserializeOwned>(&self, key: &str) -> Result<T, StorageError> {
            let path = self.key_path(key);
            let content = fs::read_to_string(&path)
                .map_err(|e| StorageError(format!("Failed to read {}: {}", path.display(), e)))?;
            serde_json::from_str(&content)
                .map_err(|e| StorageError(format!("Failed to parse {}: {}", path.display(), e)))
        }

        fn set<T: Serialize>(&self, key: &str, value: &T) -> Result<(), StorageError> {
            let path = self.key_path(key);
            let content = serde_json::to_string_pretty(value)
                .map_err(|e| StorageError(format!("Failed to serialize: {}", e)))?;
            write_private_file(&path, content.as_bytes())
                .map_err(|e| StorageError(format!("Failed to write {}: {}", path.display(), e)))
        }

        fn remove(&self, key: &str) -> Result<(), StorageError> {
            let path = self.key_path(key);
            if path.exists() {
                fs::remove_file(&path).map_err(|e| {
                    StorageError(format!("Failed to remove {}: {}", path.display(), e))
                })?;
            }
            Ok(())
        }

        fn contains_key(&self, key: &str) -> bool {
            self.key_path(key).exists()
        }
    }
}

// ============================================================================
// Platform-specific type aliases
// ============================================================================

#[cfg(target_arch = "wasm32")]
pub type PlatformStorage = web::WebStorage;

#[cfg(all(
    not(target_arch = "wasm32"),
    any(feature = "desktop", feature = "mobile")
))]
pub type PlatformStorage = native::NativeStorage;

// Fallback for builds without platform features (e.g., cargo check)
#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
pub type PlatformStorage = StubStorage;

// Stub storage for non-platform builds
#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
pub struct StubStorage;

#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
impl StubStorage {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
impl Default for StubStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(feature = "desktop"),
    not(feature = "mobile")
))]
impl StorageBackend for StubStorage {
    fn get<T: DeserializeOwned>(&self, _key: &str) -> Result<T, StorageError> {
        Err(StorageError(
            "Storage not available without platform features".to_string(),
        ))
    }

    fn set<T: Serialize>(&self, _key: &str, _value: &T) -> Result<(), StorageError> {
        Err(StorageError(
            "Storage not available without platform features".to_string(),
        ))
    }

    fn remove(&self, _key: &str) -> Result<(), StorageError> {
        Err(StorageError(
            "Storage not available without platform features".to_string(),
        ))
    }

    fn contains_key(&self, _key: &str) -> bool {
        false
    }
}

/// Create a new platform-appropriate storage backend
pub fn create_storage() -> PlatformStorage {
    PlatformStorage::new()
}

#[cfg(all(test, unix))]
mod private_file_tests {
    use super::{ensure_private_dir, write_private_file};
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    /// A fresh temp dir, removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("p2a-storage-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn settings_file_is_owner_only_and_replaces_world_readable_file() {
        let tmp = TempDir::new();
        let dir = &tmp.0;
        let path = dir.join("p2a-settings.json");
        std::fs::write(&path, "old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        write_private_file(&path, b"{\"a\":1}").unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"a\":1}");
        assert_eq!(
            std::fs::read_dir(dir).unwrap().count(),
            1,
            "temp file left behind"
        );
    }

    #[test]
    fn storage_dir_is_owner_only() {
        let tmp = TempDir::new();
        let dir = &tmp.0;
        let sub = dir.join("p2a");
        std::fs::create_dir(&sub).unwrap();
        std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o755)).unwrap();
        ensure_private_dir(&sub).unwrap();
        assert_eq!(
            std::fs::metadata(&sub).unwrap().permissions().mode() & 0o777,
            0o700
        );
        let fresh = dir.join("fresh/p2a");
        ensure_private_dir(&fresh).unwrap();
        assert_eq!(
            std::fs::metadata(&fresh).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }
}
