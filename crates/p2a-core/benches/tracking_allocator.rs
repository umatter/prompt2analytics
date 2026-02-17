//! Tracking allocator for precise memory measurement in benchmarks

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A wrapper around the system allocator that tracks allocation counts
pub struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let current = ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            // Update peak
            let mut peak = PEAK.load(Ordering::Relaxed);
            while current > peak {
                match PEAK.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

/// Reset tracking counters -- call before each benchmark
pub fn reset_tracking() {
    ALLOCATED.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
}

/// Get current allocated bytes
pub fn current_allocated() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

/// Get peak allocated bytes since last reset
pub fn peak_allocated() -> usize {
    PEAK.load(Ordering::Relaxed)
}
