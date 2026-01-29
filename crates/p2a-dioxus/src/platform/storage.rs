//! Platform-agnostic storage abstraction
//!
//! - Web: Uses localStorage via gloo_storage
//! - Native: Uses file-based storage in the user's config directory

use serde::{Serialize, de::DeserializeOwned};

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

#[cfg(not(target_arch = "wasm32"))]
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

            // Ensure directory exists
            if let Err(e) = fs::create_dir_all(&storage_dir) {
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
            fs::write(&path, content)
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

#[cfg(not(target_arch = "wasm32"))]
pub type PlatformStorage = native::NativeStorage;

/// Create a new platform-appropriate storage backend
pub fn create_storage() -> PlatformStorage {
    PlatformStorage::new()
}
