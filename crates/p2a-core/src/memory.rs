//! Memory estimation and profiling utilities.
//!
//! Provides functions to estimate memory usage of datasets and track
//! cumulative memory consumption across operations.
//!
//! # Example
//!
//! ```
//! use p2a_core::memory::{estimate_dataset_memory, format_bytes, MemoryTracker};
//! use p2a_core::Dataset;
//! use polars::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let df = df! {
//!     "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
//!     "y" => [2.1, 4.0, 5.9, 8.1, 10.0]
//! }?;
//! let dataset = Dataset::new(df);
//!
//! let mem = estimate_dataset_memory(&dataset);
//! println!("Dataset memory: {}", format_bytes(mem));
//!
//! let mut tracker = MemoryTracker::new(100); // 100 MB limit
//! tracker.add_dataset(&dataset);
//! println!("{}", tracker.status());
//! # Ok(())
//! # }
//! ```

use crate::Dataset;
use serde::Serialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Estimate the memory usage of a dataset in bytes.
///
/// This is a rough estimate based on the dataset dimensions and data types.
/// Actual memory usage may vary due to Polars internal optimizations,
/// string interning, and chunked array storage.
pub fn estimate_dataset_memory(dataset: &Dataset) -> usize {
    let df = dataset.df();
    let nrows = df.height();

    let mut total_bytes = 0usize;

    for col in df.get_columns() {
        let dtype = col.dtype();
        let bytes_per_value = match dtype {
            polars::datatypes::DataType::Boolean => 1,
            polars::datatypes::DataType::Int8 | polars::datatypes::DataType::UInt8 => 1,
            polars::datatypes::DataType::Int16 | polars::datatypes::DataType::UInt16 => 2,
            polars::datatypes::DataType::Int32 | polars::datatypes::DataType::UInt32 => 4,
            polars::datatypes::DataType::Int64 | polars::datatypes::DataType::UInt64 => 8,
            polars::datatypes::DataType::Float32 => 4,
            polars::datatypes::DataType::Float64 => 8,
            polars::datatypes::DataType::String => 24, // Average string overhead (pointer + len + capacity + small string)
            polars::datatypes::DataType::Date => 4,
            polars::datatypes::DataType::Datetime(_, _) => 8,
            polars::datatypes::DataType::Duration(_) => 8,
            polars::datatypes::DataType::Time => 8,
            polars::datatypes::DataType::Categorical(_, _) => 4, // Index into dictionary
            polars::datatypes::DataType::List(_) => 32,          // List overhead
            polars::datatypes::DataType::Struct(_) => 16,        // Struct overhead
            _ => 8, // Default estimate for unknown types
        };

        total_bytes += nrows * bytes_per_value;
    }

    // Add overhead for DataFrame structure (approximately 10%)
    total_bytes + (total_bytes / 10)
}

/// Format bytes as a human-readable string.
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Memory tracker for monitoring cumulative memory usage.
///
/// Tracks memory across multiple datasets and provides warnings
/// when approaching configured limits.
#[derive(Debug)]
pub struct MemoryTracker {
    /// Maximum allowed memory in bytes (0 = unlimited)
    max_bytes: usize,
    /// Current estimated memory usage in bytes
    current_bytes: usize,
    /// Whether a warning has been issued
    warning_issued: bool,
}

impl MemoryTracker {
    /// Create a new memory tracker with the given limit in MB.
    ///
    /// Pass 0 for no memory limit.
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            max_bytes: max_memory_mb.saturating_mul(1024 * 1024),
            current_bytes: 0,
            warning_issued: false,
        }
    }

    /// Create a memory tracker with no limit.
    pub fn unlimited() -> Self {
        Self::new(0)
    }

    /// Check if a memory limit is configured.
    pub fn has_limit(&self) -> bool {
        self.max_bytes > 0
    }

    /// Add memory usage from a dataset.
    ///
    /// Returns true if within limits, false if limit exceeded.
    pub fn add_dataset(&mut self, dataset: &Dataset) -> bool {
        let mem = estimate_dataset_memory(dataset);
        self.add_bytes(mem)
    }

    /// Add raw bytes to the tracker.
    ///
    /// Returns true if within limits, false if limit exceeded.
    pub fn add_bytes(&mut self, bytes: usize) -> bool {
        self.current_bytes = self.current_bytes.saturating_add(bytes);

        if self.max_bytes > 0 && self.current_bytes > self.max_bytes {
            if !self.warning_issued {
                self.warning_issued = true;
            }
            false
        } else {
            true
        }
    }

    /// Remove bytes from the tracker (e.g., when a dataset is dropped).
    pub fn remove_bytes(&mut self, bytes: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
        // Reset warning if we're back under the limit
        if self.max_bytes > 0 && self.current_bytes <= self.max_bytes {
            self.warning_issued = false;
        }
    }

    /// Get current memory usage estimate.
    pub fn current_usage(&self) -> usize {
        self.current_bytes
    }

    /// Get maximum allowed memory.
    pub fn max_allowed(&self) -> usize {
        self.max_bytes
    }

    /// Check if limit has been exceeded.
    pub fn limit_exceeded(&self) -> bool {
        self.max_bytes > 0 && self.current_bytes > self.max_bytes
    }

    /// Check if warning threshold has been reached (80% of limit).
    pub fn near_limit(&self) -> bool {
        if self.max_bytes == 0 {
            return false;
        }
        self.current_bytes > (self.max_bytes * 8 / 10)
    }

    /// Get usage percentage (0-100).
    pub fn usage_percent(&self) -> Option<f64> {
        if self.max_bytes == 0 {
            None
        } else {
            Some(self.current_bytes as f64 / self.max_bytes as f64 * 100.0)
        }
    }

    /// Format current status as a string.
    pub fn status(&self) -> String {
        if self.max_bytes == 0 {
            format!("Memory: {} (no limit)", format_bytes(self.current_bytes))
        } else {
            let pct = (self.current_bytes as f64 / self.max_bytes as f64 * 100.0) as usize;
            format!(
                "Memory: {} / {} ({}%)",
                format_bytes(self.current_bytes),
                format_bytes(self.max_bytes),
                pct
            )
        }
    }

    /// Reset the tracker to zero usage.
    pub fn reset(&mut self) {
        self.current_bytes = 0;
        self.warning_issued = false;
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Memory statistics for a server or application.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    /// Total estimated memory used by datasets
    pub dataset_memory_bytes: usize,
    /// Formatted dataset memory
    pub dataset_memory_formatted: String,
    /// Number of datasets tracked
    pub dataset_count: usize,
    /// Per-dataset memory breakdown
    pub datasets: Vec<DatasetMemoryInfo>,
    /// Process memory (if available)
    pub process_memory: Option<ProcessMemory>,
    /// Timestamp of the snapshot
    pub timestamp_ms: u64,
}

/// Memory information for a single dataset.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetMemoryInfo {
    /// Dataset name/ID
    pub name: String,
    /// Estimated memory in bytes
    pub memory_bytes: usize,
    /// Formatted memory string
    pub memory_formatted: String,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub columns: usize,
}

/// Process-level memory information.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessMemory {
    /// Resident set size (physical memory)
    pub rss_bytes: usize,
    /// Virtual memory size
    pub virtual_bytes: usize,
    /// Formatted RSS
    pub rss_formatted: String,
    /// Formatted virtual memory
    pub virtual_formatted: String,
}

/// Memory profiler for tracking dataset memory across a session.
///
/// This is designed for use in servers where multiple datasets may be
/// loaded and unloaded over time.
#[derive(Debug)]
pub struct MemoryProfiler {
    /// Memory usage per dataset (name -> bytes)
    dataset_memory: HashMap<String, usize>,
    /// History of memory snapshots (for trend analysis)
    history: Vec<MemorySnapshot>,
    /// Maximum history entries to keep
    max_history: usize,
    /// Start time for the profiler
    start_time: Instant,
    /// Peak memory usage observed
    peak_memory: usize,
}

/// A snapshot of memory state at a point in time.
#[derive(Debug, Clone, Serialize)]
pub struct MemorySnapshot {
    /// Time offset from profiler start in milliseconds
    pub time_offset_ms: u64,
    /// Total dataset memory at this time
    pub total_bytes: usize,
    /// Number of datasets
    pub dataset_count: usize,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    pub fn new() -> Self {
        Self {
            dataset_memory: HashMap::new(),
            history: Vec::new(),
            max_history: 1000,
            start_time: Instant::now(),
            peak_memory: 0,
        }
    }

    /// Create a profiler with custom history limit.
    pub fn with_history_limit(max_history: usize) -> Self {
        Self {
            max_history,
            ..Self::new()
        }
    }

    /// Track a dataset being loaded.
    pub fn track_dataset(&mut self, name: &str, dataset: &Dataset) {
        let mem = estimate_dataset_memory(dataset);
        self.dataset_memory.insert(name.to_string(), mem);
        self.update_peak();
        self.record_snapshot();
    }

    /// Track a dataset being unloaded.
    pub fn untrack_dataset(&mut self, name: &str) {
        self.dataset_memory.remove(name);
        self.record_snapshot();
    }

    /// Get memory for a specific dataset.
    pub fn dataset_memory(&self, name: &str) -> Option<usize> {
        self.dataset_memory.get(name).copied()
    }

    /// Get total memory across all datasets.
    pub fn total_memory(&self) -> usize {
        self.dataset_memory.values().sum()
    }

    /// Get peak memory observed.
    pub fn peak_memory(&self) -> usize {
        self.peak_memory
    }

    /// Get number of tracked datasets.
    pub fn dataset_count(&self) -> usize {
        self.dataset_memory.len()
    }

    /// Get elapsed time since profiler creation.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get memory statistics snapshot.
    pub fn stats(&self) -> MemoryStats {
        let total = self.total_memory();
        let datasets: Vec<DatasetMemoryInfo> = self
            .dataset_memory
            .iter()
            .map(|(name, &bytes)| DatasetMemoryInfo {
                name: name.clone(),
                memory_bytes: bytes,
                memory_formatted: format_bytes(bytes),
                rows: 0,    // Would need dataset access to fill
                columns: 0, // Would need dataset access to fill
            })
            .collect();

        MemoryStats {
            dataset_memory_bytes: total,
            dataset_memory_formatted: format_bytes(total),
            dataset_count: datasets.len(),
            datasets,
            process_memory: get_process_memory(),
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }

    /// Get detailed statistics including dataset dimensions.
    pub fn stats_with_datasets<'a>(
        &self,
        datasets: impl Iterator<Item = (&'a str, &'a Dataset)>,
    ) -> MemoryStats {
        let mut dataset_info: Vec<DatasetMemoryInfo> = Vec::new();
        let mut total = 0usize;

        for (name, dataset) in datasets {
            let mem = estimate_dataset_memory(dataset);
            total += mem;
            dataset_info.push(DatasetMemoryInfo {
                name: name.to_string(),
                memory_bytes: mem,
                memory_formatted: format_bytes(mem),
                rows: dataset.nrows(),
                columns: dataset.ncols(),
            });
        }

        // Sort by memory usage (largest first)
        dataset_info.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes));

        MemoryStats {
            dataset_memory_bytes: total,
            dataset_memory_formatted: format_bytes(total),
            dataset_count: dataset_info.len(),
            datasets: dataset_info,
            process_memory: get_process_memory(),
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }

    /// Get memory history for trend analysis.
    pub fn history(&self) -> &[MemorySnapshot] {
        &self.history
    }

    /// Clear history while keeping current tracking.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    fn update_peak(&mut self) {
        let total = self.total_memory();
        if total > self.peak_memory {
            self.peak_memory = total;
        }
    }

    fn record_snapshot(&mut self) {
        let snapshot = MemorySnapshot {
            time_offset_ms: self.start_time.elapsed().as_millis() as u64,
            total_bytes: self.total_memory(),
            dataset_count: self.dataset_count(),
        };

        self.history.push(snapshot);

        // Trim history if needed
        if self.history.len() > self.max_history {
            let excess = self.history.len() - self.max_history;
            self.history.drain(0..excess);
        }
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Get process memory information.
///
/// This uses platform-specific APIs to get actual process memory usage.
/// Returns None if the information is not available.
#[cfg(target_os = "linux")]
pub fn get_process_memory() -> Option<ProcessMemory> {
    use std::fs;

    // Read /proc/self/statm for memory info
    // Format: size resident shared text lib data dt
    // Units are pages (usually 4KB)
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let parts: Vec<&str> = statm.split_whitespace().collect();

    if parts.len() >= 2 {
        let page_size = 4096usize; // Typical page size
        let virtual_pages: usize = parts[0].parse().ok()?;
        let rss_pages: usize = parts[1].parse().ok()?;

        let virtual_bytes = virtual_pages * page_size;
        let rss_bytes = rss_pages * page_size;

        Some(ProcessMemory {
            rss_bytes,
            virtual_bytes,
            rss_formatted: format_bytes(rss_bytes),
            virtual_formatted: format_bytes(virtual_bytes),
        })
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
pub fn get_process_memory() -> Option<ProcessMemory> {
    // macOS uses different APIs - would need mach APIs
    // For now, return None (could be implemented with mach_task_basic_info)
    None
}

#[cfg(target_os = "windows")]
pub fn get_process_memory() -> Option<ProcessMemory> {
    // Windows uses different APIs - would need GetProcessMemoryInfo
    // For now, return None
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub fn get_process_memory() -> Option<ProcessMemory> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn make_test_dataset() -> Dataset {
        let df = df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "y" => [2.1, 4.0, 5.9, 8.1, 10.0],
            "name" => ["a", "b", "c", "d", "e"]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.00 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_estimate_dataset_memory() {
        let dataset = make_test_dataset();
        let mem = estimate_dataset_memory(&dataset);
        // Should be > 0 and reasonable for 5 rows x 3 cols
        assert!(mem > 0);
        assert!(mem < 10_000); // Should be less than 10KB for this tiny dataset
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new(100); // 100 MB limit
        assert!(tracker.has_limit());
        assert_eq!(tracker.max_allowed(), 100 * 1024 * 1024);

        let dataset = make_test_dataset();
        assert!(tracker.add_dataset(&dataset));
        assert!(tracker.current_usage() > 0);
        assert!(!tracker.limit_exceeded());
    }

    #[test]
    fn test_memory_tracker_unlimited() {
        let tracker = MemoryTracker::unlimited();
        assert!(!tracker.has_limit());
        assert_eq!(tracker.max_allowed(), 0);
        assert!(tracker.usage_percent().is_none());
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();
        let dataset = make_test_dataset();

        profiler.track_dataset("test1", &dataset);
        assert_eq!(profiler.dataset_count(), 1);
        assert!(profiler.total_memory() > 0);

        profiler.track_dataset("test2", &dataset);
        assert_eq!(profiler.dataset_count(), 2);

        let stats = profiler.stats();
        assert_eq!(stats.dataset_count, 2);

        profiler.untrack_dataset("test1");
        assert_eq!(profiler.dataset_count(), 1);
    }

    #[test]
    fn test_memory_profiler_history() {
        let mut profiler = MemoryProfiler::with_history_limit(10);
        let dataset = make_test_dataset();

        for i in 0..15 {
            profiler.track_dataset(&format!("ds{}", i), &dataset);
        }

        // History should be limited to 10 entries
        assert!(profiler.history().len() <= 10);
    }

    #[test]
    fn test_memory_tracker_near_limit() {
        let mut tracker = MemoryTracker::new(1); // 1 MB limit

        // Add bytes until near limit
        tracker.add_bytes(900_000); // 900 KB
        assert!(tracker.near_limit());

        // Add more to exceed
        tracker.add_bytes(200_000); // Now over 1 MB
        assert!(tracker.limit_exceeded());
    }

    #[test]
    fn test_memory_tracker_remove_bytes() {
        let mut tracker = MemoryTracker::new(1); // 1 MB limit

        tracker.add_bytes(1_500_000); // 1.5 MB - over limit
        assert!(tracker.limit_exceeded());

        tracker.remove_bytes(600_000); // Back to 900 KB
        assert!(!tracker.limit_exceeded());
    }
}
