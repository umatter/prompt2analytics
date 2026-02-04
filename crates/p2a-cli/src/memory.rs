//! Memory estimation and monitoring utilities (re-exported from p2a-core).
//!
//! This module re-exports memory utilities from p2a-core for backward compatibility.

pub use p2a_core::memory::{
    DatasetMemoryInfo, MemoryProfiler, MemorySnapshot, MemoryStats, MemoryTracker, ProcessMemory,
    estimate_dataset_memory, format_bytes, get_process_memory,
};
