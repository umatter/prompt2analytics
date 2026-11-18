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

/// Bytes of one String or Binary view (polars' `View`), charged per value on top of the frame's
/// buffer sizes, as the MCP server's session accounting charges it.
const VIEW_BYTES: usize = 16;

/// Estimate the memory usage of a dataset in bytes: polars' `DataFrame::estimated_size` (the
/// sizes of the frame's buffers, text bytes included) plus 16 bytes per String or Binary value
/// (its view).
pub fn estimate_dataset_memory(dataset: &Dataset) -> usize {
    let df = dataset.df();
    let views: usize = df
        .get_columns()
        .iter()
        .filter(|c| {
            matches!(
                c.dtype(),
                polars::datatypes::DataType::String | polars::datatypes::DataType::Binary
            )
        })
        .map(|c| c.len())
        .sum();
    df.estimated_size()
        .saturating_add(views.saturating_mul(VIEW_BYTES))
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
/// Tracks the estimated memory of several datasets against a limit. It prints nothing
/// itself: [`MemoryTracker::take_warning`] formats one warning per crossing of the limit,
/// which the caller reports.
#[derive(Debug)]
pub struct MemoryTracker {
    /// Maximum allowed memory in bytes (0 = unlimited)
    max_bytes: usize,
    /// Current estimated memory usage in bytes
    current_bytes: usize,
    /// Whether the warning for the current crossing has been taken
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
        !self.limit_exceeded()
    }

    /// "estimated dataset memory 1.43 MB exceeds the limit of 1.00 MB" the first time usage
    /// exceeds the limit since creation, [`reset`](Self::reset), or usage last fell back to the
    /// limit; `None` otherwise and with no limit.
    pub fn take_warning(&mut self) -> Option<String> {
        if !self.limit_exceeded() || self.warning_issued {
            return None;
        }
        self.warning_issued = true;
        Some(format!(
            "estimated dataset memory {} exceeds the limit of {}",
            format_bytes(self.current_bytes),
            format_bytes(self.max_bytes)
        ))
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
        // max_bytes / 10 * 8 cannot overflow, unlike max_bytes * 8 / 10
        self.current_bytes > self.max_bytes / 10 * 8
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

/// The `AT_PAGESZ` (key 6) entry of an auxiliary vector (`/proc/self/auxv`: native-endian
/// (key, value) pairs of machine words, ending at key 0); `None` without one.
#[cfg(any(target_os = "linux", test))]
fn auxv_page_size(auxv: &[u8]) -> Option<usize> {
    const AT_PAGESZ: usize = 6;
    let word = std::mem::size_of::<usize>();
    let read = |chunk: &[u8]| -> usize {
        let mut b = [0u8; std::mem::size_of::<usize>()];
        b.copy_from_slice(chunk);
        usize::from_ne_bytes(b)
    };
    for pair in auxv.chunks_exact(2 * word) {
        let (key, value) = (read(&pair[..word]), read(&pair[word..]));
        match key {
            0 => return None,
            AT_PAGESZ if value > 0 => return Some(value),
            _ => {}
        }
    }
    None
}

/// The kernel's page size, read once from `/proc/self/auxv`; 4096 when it cannot be read.
#[cfg(target_os = "linux")]
fn page_size() -> usize {
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        std::fs::read("/proc/self/auxv")
            .ok()
            .and_then(|auxv| auxv_page_size(&auxv))
            .unwrap_or(4096)
    })
}

/// Get process memory information.
///
/// Linux only (`/proc/self/statm`, in pages of the kernel's page size); `None` elsewhere, and
/// when the information cannot be read.
#[cfg(target_os = "linux")]
pub fn get_process_memory() -> Option<ProcessMemory> {
    use std::fs;

    // Read /proc/self/statm for memory info
    // Format: size resident shared text lib data dt, in pages
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let parts: Vec<&str> = statm.split_whitespace().collect();

    if parts.len() >= 2 {
        let page_size = page_size();
        let virtual_pages: usize = parts[0].parse().ok()?;
        let rss_pages: usize = parts[1].parse().ok()?;

        let virtual_bytes = virtual_pages.saturating_mul(page_size);
        let rss_bytes = rss_pages.saturating_mul(page_size);

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

/// Process memory is read on Linux only: `None` here.
#[cfg(target_os = "macos")]
pub fn get_process_memory() -> Option<ProcessMemory> {
    // macOS uses different APIs - would need mach APIs
    // For now, return None (could be implemented with mach_task_basic_info)
    None
}

/// Process memory is read on Linux only: `None` here.
#[cfg(target_os = "windows")]
pub fn get_process_memory() -> Option<ProcessMemory> {
    // Windows uses different APIs - would need GetProcessMemoryInfo
    // For now, return None
    None
}

/// Process memory is read on Linux only: `None` here.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub fn get_process_memory() -> Option<ProcessMemory> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn estimate_tracks_estimated_size() {
        // Pre-fix a 40-byte text value was charged 24 bytes plus 10% (26.4).
        let n = 1_000;
        let text: Vec<String> = (0..n).map(|i| format!("{i:0>40}")).collect();
        let df = df! {
            "f" => (0..n).map(|i| i as f64 * 0.5 + 0.1).collect::<Vec<_>>(),
            "i" => (0..n).map(|i| i as i32).collect::<Vec<_>>(),
            "s" => text,
            "b" => (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>(),
        }
        .unwrap();
        let ds = Dataset::new(df.clone());
        let est = estimate_dataset_memory(&ds);
        assert_eq!(est, df.estimated_size() + 16 * n);
        let s_only = Dataset::new(df.select(["s"]).unwrap());
        assert!(estimate_dataset_memory(&s_only) >= 56 * n);
    }

    #[test]
    fn auxv_parser_finds_the_page_size_or_none() {
        let w = std::mem::size_of::<usize>();
        let pairs = |kv: &[(usize, usize)]| -> Vec<u8> {
            kv.iter()
                .flat_map(|(k, v)| {
                    let mut b = k.to_ne_bytes().to_vec();
                    b.extend(v.to_ne_bytes());
                    b
                })
                .collect()
        };
        assert_eq!(
            auxv_page_size(&pairs(&[(3, 7), (6, 16384), (0, 0)])),
            Some(16384)
        );
        assert_eq!(auxv_page_size(&pairs(&[(3, 7), (0, 0), (6, 4096)])), None);
        assert_eq!(auxv_page_size(&pairs(&[(3, 7)])), None);
        assert_eq!(auxv_page_size(&[0u8; 3]), None);
        assert_eq!(pairs(&[(1, 2)]).len(), 2 * w);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn page_size_comes_from_auxv() {
        let auxv = std::fs::read("/proc/self/auxv").unwrap();
        let w = std::mem::size_of::<usize>();
        let word = |b: &[u8]| usize::from_ne_bytes(b.try_into().unwrap());
        let expected = auxv
            .chunks_exact(2 * w)
            .find(|p| word(&p[..w]) == 6)
            .map(|p| word(&p[w..]))
            .unwrap();
        assert_eq!(page_size(), expected);
        assert!(expected >= 4096 && expected.is_power_of_two());
    }

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
    fn test_near_limit_does_not_overflow_at_usize_max() {
        let mut t = MemoryTracker::new(usize::MAX); // saturates to usize::MAX bytes
        assert_eq!(t.max_allowed(), usize::MAX);
        assert!(!t.near_limit()); // panicked before: max_bytes * 8 overflows
        t.add_bytes(usize::MAX / 10 * 8 + 1);
        assert!(t.near_limit() && !t.limit_exceeded());
    }

    #[test]
    fn test_take_warning_once_per_crossing() {
        let mut t = MemoryTracker::new(1);
        assert_eq!(t.take_warning(), None);
        assert!(!t.add_bytes(1_500_000));
        assert_eq!(
            t.take_warning().as_deref(),
            Some("estimated dataset memory 1.43 MB exceeds the limit of 1.00 MB")
        );
        assert_eq!(t.take_warning(), None);
        t.add_bytes(1);
        assert_eq!(t.take_warning(), None);
        t.remove_bytes(1_000_000); // 500,001 bytes: under the limit again
        assert_eq!(t.take_warning(), None);
        t.add_bytes(600_000); // 1,100,001 bytes: a new crossing
        assert_eq!(
            t.take_warning().as_deref(),
            Some("estimated dataset memory 1.05 MB exceeds the limit of 1.00 MB")
        );
        let mut u = MemoryTracker::unlimited();
        u.add_bytes(usize::MAX);
        assert_eq!(u.take_warning(), None);
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
