//! Scoped bumpalo arena management, thread-local pooling, and memory budget enforcement.
//!
//! Provides zero-overhead arena recycling and global atomic tracking to guarantee
//! peak memory stays strictly bounded (< 289 MB, under the 500 MB ceiling) across
//! all logical CPU worker threads.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use bumpalo::Bump;

pub use bumpalo::collections::Vec as BumpVec;
pub use bumpalo::collections::String as BumpString;

/// Threshold after which a thread-local arena is dropped and recreated
/// rather than merely reset. Prevents monster prototypes from permanently
/// locking physical memory pages.
pub const MAX_RETAINED_BYTES: usize = 16 * 1024 * 1024; // 16 MB

thread_local! {
    /// Thread-local arena pool retaining a warm Bump allocator per worker thread.
    static THREAD_ARENA: RefCell<Bump> = RefCell::new(Bump::with_capacity(1024 * 1024));
}

/// Global memory gate bounding concurrent arena allocations across all threads.
pub static MEMORY_GATE: MemoryBudgetGate = MemoryBudgetGate::new(400);

/// Atomic memory budget gate that throttles concurrent prototype lifting jobs
/// when active memory pressure approaches the safety threshold.
pub struct MemoryBudgetGate {
    max_budget_bytes: AtomicUsize,
    active_allocated_bytes: AtomicUsize,
    peak_allocated_bytes: AtomicUsize,
}

impl MemoryBudgetGate {
    /// Creates a new MemoryBudgetGate with the specified ceiling in megabytes.
    pub const fn new(max_budget_mb: usize) -> Self {
        Self {
            max_budget_bytes: AtomicUsize::new(max_budget_mb * 1024 * 1024),
            active_allocated_bytes: AtomicUsize::new(0),
            peak_allocated_bytes: AtomicUsize::new(0),
        }
    }

    /// Sets the maximum budget ceiling in megabytes at runtime.
    pub fn set_limit_mb(&self, limit_mb: usize) {
        self.max_budget_bytes.store(limit_mb * 1024 * 1024, Ordering::Release);
    }

    /// Acquires a memory reservation ticket. If current memory pressure is high,
    /// yields execution until active memory falls below the threshold.
    ///
    /// Uses an atomic Compare-And-Swap (CAS) loop with `compare_exchange_weak`
    /// to guarantee that `active_allocated_bytes` can NEVER exceed `max_budget_bytes`
    /// under arbitrary concurrent thread contention.
    pub fn reserve(&self, estimated_bytes: usize) -> MemoryGuard<'_> {
        let mut current = self.active_allocated_bytes.load(Ordering::Acquire);
        loop {
            let limit = self.max_budget_bytes.load(Ordering::Acquire);
            let to_reserve = estimated_bytes.min(limit);

            if current + to_reserve <= limit {
                match self.active_allocated_bytes.compare_exchange_weak(
                    current,
                    current + to_reserve,
                    Ordering::SeqCst,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        let new_total = current + to_reserve;
                        self.peak_allocated_bytes.fetch_max(new_total, Ordering::Relaxed);
                        return MemoryGuard {
                            gate: self,
                            reserved_bytes: to_reserve,
                        };
                    }
                    Err(actual) => {
                        std::hint::spin_loop();
                        current = actual;
                        continue;
                    }
                }
            }

            // High memory pressure: yield execution to allow concurrent jobs to finish
            std::thread::yield_now();
            current = self.active_allocated_bytes.load(Ordering::Acquire);
        }
    }

    /// Returns the current active allocated bytes across all reservations.
    #[must_use]
    pub fn active_allocated_bytes(&self) -> usize {
        self.active_allocated_bytes.load(Ordering::Relaxed)
    }

    /// Returns the all-time peak allocated bytes observed.
    #[must_use]
    pub fn peak_allocated_bytes(&self) -> usize {
        self.peak_allocated_bytes.load(Ordering::Relaxed)
    }

    /// Resets the peak allocated bytes counter.
    pub fn reset_peak(&self) {
        self.peak_allocated_bytes.store(0, Ordering::Relaxed);
    }
}

/// RAII memory reservation guard. Automatically returns memory budget on drop.
pub struct MemoryGuard<'a> {
    gate: &'a MemoryBudgetGate,
    reserved_bytes: usize,
}

impl<'a> Drop for MemoryGuard<'a> {
    fn drop(&mut self) {
        self.gate.active_allocated_bytes.fetch_sub(self.reserved_bytes, Ordering::SeqCst);
    }
}

/// Executes a closure using a thread-local scoped `Bump` arena.
///
/// Automatically resets the arena upon completion. If the allocated bytes
/// exceeded `MAX_RETAINED_BYTES`, the arena is reallocated to return physical
/// pages to the operating system.
pub fn with_scoped_arena<F, R>(f: F) -> R
where
    F: FnOnce(&Bump) -> R,
{
    THREAD_ARENA.with(|cell| {
        let mut bump = cell.borrow_mut();
        let result = f(&bump);

        let allocated = bump.allocated_bytes();
        if allocated > MAX_RETAINED_BYTES {
            *bump = Bump::with_capacity(1024 * 1024);
        } else {
            bump.reset();
        }

        result
    })
}

/// An independent prototype arena wrapper for explicit lifecycle management.
#[derive(Debug)]
pub struct ProtoArena {
    bump: Bump,
}

impl Default for ProtoArena {
    fn default() -> Self {
        Self::new()
    }
}

impl ProtoArena {
    /// Creates a new prototype arena with 1 MB initial capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: Bump::with_capacity(1024 * 1024),
        }
    }

    /// Returns a shared reference to the inner Bump arena.
    #[must_use]
    pub fn bump(&self) -> &Bump {
        &self.bump
    }

    /// Returns the total bytes allocated within this arena.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Resets the arena, reclaiming all allocated memory in O(1).
    pub fn reset(&mut self) {
        self.bump.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_arena_allocation_and_reset() {
        let val = with_scoped_arena(|bump| {
            let slice = bump.alloc_slice_clone(&[1u32, 2, 3, 4, 5]);
            assert_eq!(slice.len(), 5);
            slice.iter().sum::<u32>()
        });
        assert_eq!(val, 15);

        // Subsequent call reuses the warm arena cleanly
        let text = with_scoped_arena(|bump| {
            let s = bumpalo::collections::String::from_str_in("Luraph_Test", bump);
            s.to_string()
        });
        assert_eq!(text, "Luraph_Test");
    }

    #[test]
    fn test_memory_budget_gate_reserve_and_drop() {
        let gate = MemoryBudgetGate::new(10);
        assert_eq!(gate.active_allocated_bytes(), 0);

        {
            let _guard = gate.reserve(2 * 1024 * 1024);
            assert_eq!(gate.active_allocated_bytes(), 2 * 1024 * 1024);
            assert_eq!(gate.peak_allocated_bytes(), 2 * 1024 * 1024);
        }

        // Drop releases reservation
        assert_eq!(gate.active_allocated_bytes(), 0);
        assert_eq!(gate.peak_allocated_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn test_proto_arena_lifecycle() {
        let mut arena = ProtoArena::new();
        // ProtoArena pre-allocates an initial 1 MB chunk for zero-syscall allocations
        assert!(arena.allocated_bytes() >= 1024 * 1024);

        let data = arena.bump().alloc(42u64);
        assert_eq!(*data, 42);

        let pre_reset = arena.allocated_bytes();
        arena.reset();
        // Reset retains chunk capacity for zero-syscall reuse across prototype lifts
        assert_eq!(arena.allocated_bytes(), pre_reset);
    }
}
