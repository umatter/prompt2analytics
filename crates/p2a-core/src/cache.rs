//! Result caching for expensive operations.
//!
//! Provides an LRU cache for storing and reusing results from repeated operations
//! like regressions, statistics computations, and other analyses.
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::cache::{ResultCache, CacheKey};
//!
//! let mut cache = ResultCache::new(100); // 100 entry capacity
//!
//! // Create a cache key from operation parameters
//! let key = CacheKey::new("ols")
//!     .with_param("dataset", "mydata")
//!     .with_param("y", "price")
//!     .with_params("x", &["sqft", "bedrooms"]);
//!
//! // Check if result is cached
//! if let Some(result) = cache.get::<OlsResult>(&key) {
//!     // Use cached result
//! } else {
//!     // Compute result and cache it
//!     let result = run_ols(...)?;
//!     cache.insert(&key, &result);
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

/// A cache key built from operation name and parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// Operation name (e.g., "ols", "t_test", "kmeans")
    operation: String,
    /// Sorted parameters for consistent hashing
    params: Vec<(String, String)>,
}

impl CacheKey {
    /// Create a new cache key for an operation.
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            params: Vec::new(),
        }
    }

    /// Add a single parameter to the key.
    pub fn with_param(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.push((name.into(), value.into()));
        self.params.sort_by(|a, b| a.0.cmp(&b.0));
        self
    }

    /// Add multiple values for a parameter (e.g., list of columns).
    pub fn with_params(mut self, name: impl Into<String>, values: &[impl AsRef<str>]) -> Self {
        let name = name.into();
        let combined: String = values
            .iter()
            .map(|v| v.as_ref())
            .collect::<Vec<_>>()
            .join(",");
        self.params.push((name, combined));
        self.params.sort_by(|a, b| a.0.cmp(&b.0));
        self
    }

    /// Add a numeric parameter.
    pub fn with_num<T: std::fmt::Display>(self, name: impl Into<String>, value: T) -> Self {
        self.with_param(name, value.to_string())
    }

    /// Add a boolean parameter.
    pub fn with_bool(self, name: impl Into<String>, value: bool) -> Self {
        self.with_param(name, if value { "true" } else { "false" })
    }

    /// Get a string representation for display.
    pub fn display(&self) -> String {
        let params: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        if params.is_empty() {
            self.operation.clone()
        } else {
            format!("{}({})", self.operation, params.join(", "))
        }
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operation.hash(state);
        for (k, v) in &self.params {
            k.hash(state);
            v.hash(state);
        }
    }
}

/// A cached entry with metadata.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Serialized result data (JSON)
    data: String,
    /// When this entry was created
    created_at: Instant,
    /// Number of times this entry has been accessed
    access_count: usize,
    /// Last access time
    last_accessed: Instant,
}

impl CacheEntry {
    fn new(data: String) -> Self {
        let now = Instant::now();
        Self {
            data,
            created_at: now,
            access_count: 0,
            last_accessed: now,
        }
    }

    fn touch(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }

    fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// LRU cache for operation results.
pub struct ResultCache {
    /// Maximum number of entries
    capacity: usize,
    /// Cached entries
    entries: HashMap<CacheKey, CacheEntry>,
    /// Cache hit count
    hits: usize,
    /// Cache miss count
    misses: usize,
    /// Maximum age for entries (None = no expiration)
    max_age: Option<Duration>,
}

impl ResultCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
            max_age: None,
        }
    }

    /// Create a cache with a maximum entry age.
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Get a cached result if it exists and hasn't expired.
    pub fn get<T: for<'de> Deserialize<'de>>(&mut self, key: &CacheKey) -> Option<T> {
        // Check if entry exists and hasn't expired
        if let Some(entry) = self.entries.get_mut(key) {
            // Check expiration
            if let Some(max_age) = self.max_age {
                if entry.age() > max_age {
                    self.entries.remove(key);
                    self.misses += 1;
                    return None;
                }
            }

            entry.touch();
            self.hits += 1;

            // Deserialize the cached data
            match serde_json::from_str(&entry.data) {
                Ok(result) => Some(result),
                Err(_) => {
                    // Invalid cached data, remove it
                    self.entries.remove(key);
                    None
                }
            }
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a result into the cache.
    pub fn insert<T: Serialize>(&mut self, key: &CacheKey, value: &T) -> bool {
        // Serialize the value
        let data = match serde_json::to_string(value) {
            Ok(d) => d,
            Err(_) => return false,
        };

        // Evict if at capacity
        if self.entries.len() >= self.capacity && !self.entries.contains_key(key) {
            self.evict_lru();
        }

        self.entries.insert(key.clone(), CacheEntry::new(data));
        true
    }

    /// Check if a key exists in the cache (without updating access time).
    pub fn contains(&self, key: &CacheKey) -> bool {
        if let Some(entry) = self.entries.get(key) {
            if let Some(max_age) = self.max_age {
                return entry.age() <= max_age;
            }
            true
        } else {
            false
        }
    }

    /// Remove an entry from the cache.
    pub fn remove(&mut self, key: &CacheKey) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Clear all entries from the cache.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Invalidate all entries for a specific operation.
    pub fn invalidate_operation(&mut self, operation: &str) {
        self.entries
            .retain(|key, _| key.operation != operation);
    }

    /// Invalidate all entries that reference a specific dataset.
    pub fn invalidate_dataset(&mut self, dataset_name: &str) {
        self.entries.retain(|key, _| {
            !key.params
                .iter()
                .any(|(k, v)| k == "dataset" && v == dataset_name)
        });
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total_requests = self.hits + self.misses;
        let hit_rate = if total_requests > 0 {
            self.hits as f64 / total_requests as f64
        } else {
            0.0
        };

        CacheStats {
            capacity: self.capacity,
            size: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate,
        }
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone())
        {
            self.entries.remove(&lru_key);
        }
    }

    /// Remove all expired entries.
    pub fn cleanup_expired(&mut self) {
        if let Some(max_age) = self.max_age {
            self.entries.retain(|_, entry| entry.age() <= max_age);
        }
    }
}

impl Default for ResultCache {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    /// Maximum capacity
    pub capacity: usize,
    /// Current number of entries
    pub size: usize,
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache: {}/{} entries, {:.1}% hit rate ({} hits, {} misses)",
            self.size,
            self.capacity,
            self.hit_rate * 100.0,
            self.hits,
            self.misses
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestResult {
        value: f64,
        name: String,
    }

    #[test]
    fn test_cache_basic() {
        let mut cache = ResultCache::new(10);

        let key = CacheKey::new("test")
            .with_param("dataset", "mydata")
            .with_param("col", "x");

        let result = TestResult {
            value: 42.0,
            name: "test".to_string(),
        };

        // Insert and retrieve
        assert!(cache.insert(&key, &result));
        let cached: Option<TestResult> = cache.get(&key);
        assert_eq!(cached, Some(result));

        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn test_cache_key_ordering() {
        // Keys with same params in different order should be equal
        let key1 = CacheKey::new("ols")
            .with_param("dataset", "data")
            .with_param("y", "price");

        let key2 = CacheKey::new("ols")
            .with_param("y", "price")
            .with_param("dataset", "data");

        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = ResultCache::new(2);

        let key1 = CacheKey::new("op1");
        let key2 = CacheKey::new("op2");
        let key3 = CacheKey::new("op3");

        let result = TestResult {
            value: 1.0,
            name: "test".to_string(),
        };

        cache.insert(&key1, &result);
        cache.insert(&key2, &result);

        // Access key1 to make it more recently used
        let _: Option<TestResult> = cache.get(&key1);

        // Insert key3, should evict key2 (LRU)
        cache.insert(&key3, &result);

        assert!(cache.contains(&key1));
        assert!(!cache.contains(&key2));
        assert!(cache.contains(&key3));
    }

    #[test]
    fn test_cache_invalidation() {
        let mut cache = ResultCache::new(10);

        let key1 = CacheKey::new("ols").with_param("dataset", "data1");
        let key2 = CacheKey::new("ols").with_param("dataset", "data2");
        let key3 = CacheKey::new("ttest").with_param("dataset", "data1");

        let result = TestResult {
            value: 1.0,
            name: "test".to_string(),
        };

        cache.insert(&key1, &result);
        cache.insert(&key2, &result);
        cache.insert(&key3, &result);

        // Invalidate by dataset
        cache.invalidate_dataset("data1");

        assert!(!cache.contains(&key1));
        assert!(cache.contains(&key2));
        assert!(!cache.contains(&key3));
    }
}
