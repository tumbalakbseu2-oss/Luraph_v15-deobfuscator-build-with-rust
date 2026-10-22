//! Adversarial stress test suite for MemoryBudgetGate, THREAD_ARENA, and prototype churn.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use luraph_deobf_rs::memory::{with_scoped_arena, MemoryBudgetGate, MEMORY_GATE};

#[repr(C)]
#[allow(non_snake_case)]
struct PROCESS_MEMORY_COUNTERS {
    cb: u32,
    PageFaultCount: u32,
    PeakWorkingSetSize: usize,
    WorkingSetSize: usize,
    QuotaPeakPagedPoolUsage: usize,
    QuotaPagedPoolUsage: usize,
    QuotaPeakNonPagedPoolUsage: usize,
    QuotaNonPagedPoolUsage: usize,
    PagefileUsage: usize,
    PeakPagefileUsage: usize,
}

#[link(name = "psapi")]
extern "system" {
    fn GetCurrentProcess() -> *mut std::ffi::c_void;
    fn K32GetProcessMemoryInfo(
        process: *mut std::ffi::c_void,
        ppsmemCounters: *mut PROCESS_MEMORY_COUNTERS,
        cb: u32,
    ) -> i32;
}

fn get_process_memory_mb() -> (f64, f64) {
    let mut pmc: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let success = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
    if success != 0 {
        let ws_mb = pmc.WorkingSetSize as f64 / (1024.0 * 1024.0);
        let commit_mb = pmc.PagefileUsage as f64 / (1024.0 * 1024.0);
        (ws_mb, commit_mb)
    } else {
        (0.0, 0.0)
    }
}

/// Stress test 1: MemoryBudgetGate concurrency, contention, limit enforcement, and deadlock freedom.
#[test]
fn test_adversarial_memory_gate_concurrency_and_limits() {
    let budget_mb = 289;
    let budget_bytes = budget_mb * 1024 * 1024;
    let gate = Arc::new(MemoryBudgetGate::new(budget_mb));

    let num_threads = 16;
    let request_per_thread = 30 * 1024 * 1024; // 16 * 30 MB = 480 MB > 289 MB
    let barrier = Arc::new(Barrier::new(num_threads));
    let violated_limit = Arc::new(AtomicBool::new(false));
    let max_active_observed = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    for _ in 0..num_threads {
        let gate_clone = Arc::clone(&gate);
        let barrier_clone = Arc::clone(&barrier);
        let violated_clone = Arc::clone(&violated_limit);
        let max_active_clone = Arc::clone(&max_active_observed);

        handles.push(thread::spawn(move || {
            // Synchronize all threads to start reservation at the exact same instant
            barrier_clone.wait();

            for _ in 0..50 {
                let _guard = gate_clone.reserve(request_per_thread);

                let active = gate_clone.active_allocated_bytes();
                max_active_clone.fetch_max(active, Ordering::SeqCst);

                if active > budget_bytes {
                    violated_clone.store(true, Ordering::SeqCst);
                }

                // Hold reservation for a tiny slice
                thread::yield_now();
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join failed");
    }

    let peak = gate.peak_allocated_bytes();
    let max_observed = max_active_observed.load(Ordering::SeqCst);
    let peak_mb = peak as f64 / (1024.0 * 1024.0);
    let max_mb = max_observed as f64 / (1024.0 * 1024.0);

    println!(
        "[EMPIRICAL] MemoryBudgetGate peak: {:.2} MB, max active observed: {:.2} MB (Budget: {} MB)",
        peak_mb, max_mb, budget_mb
    );

    // Verify all reservations were properly returned on drop
    assert_eq!(gate.active_allocated_bytes(), 0, "All reservations must return to 0");

    // Check whether the gate held under concurrent contention or leaked past the 289 MB ceiling
    let exceeded = violated_limit.load(Ordering::SeqCst) || peak > budget_bytes;
    if exceeded {
        println!(
            "[FAILURE/OBSERVATION] MemoryBudgetGate allowed concurrent over-allocation: peak was {:.2} MB > {} MB budget!",
            peak_mb, budget_mb
        );
    }
    assert!(
        !exceeded,
        "MemoryBudgetGate FAILED: peak allocated ({:.2} MB) exceeded budget limit ({} MB) under 16-thread contention!",
        peak_mb, budget_mb
    );
}

/// Stress test 1b: Precise synchronized TOCTOU race condition in MemoryBudgetGate.
/// Tests whether simultaneous checks allow total reservations to blow past the budget limit.
#[test]
fn test_adversarial_memory_gate_toctou_oversubscription() {
    let budget_mb = 289;
    let budget_bytes = budget_mb * 1024 * 1024;
    let gate = Arc::new(MemoryBudgetGate::new(budget_mb));

    // Run multiple attempts to provoke the race window between check and fetch_add
    let mut race_detected = false;
    let mut max_observed_peak = 0;

    for _trial in 0..100 {
        let num_threads = 4;
        let request_bytes = 100 * 1024 * 1024; // 4 * 100 MB = 400 MB > 289 MB
        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = Vec::new();

        for _ in 0..num_threads {
            let gate_clone = Arc::clone(&gate);
            let barrier_clone = Arc::clone(&barrier);

            handles.push(thread::spawn(move || {
                barrier_clone.wait();
                let _guard = gate_clone.reserve(request_bytes);
                // Keep reservation alive until all contenders have attempted
                thread::sleep(Duration::from_millis(2));
            }));
        }

        for h in handles {
            h.join().expect("join");
        }

        let peak = gate.peak_allocated_bytes();
        if peak > max_observed_peak {
            max_observed_peak = peak;
        }
        if peak > budget_bytes {
            race_detected = true;
            println!(
                "[VULNERABILITY REPRODUCED] Trial {} provoked TOCTOU race: Peak allocated was {:.2} MB > {} MB limit!",
                _trial,
                peak as f64 / (1024.0 * 1024.0),
                budget_mb
            );
            break;
        }
        gate.reset_peak();
    }

    let max_mb = max_observed_peak as f64 / (1024.0 * 1024.0);
    println!(
        "[EMPIRICAL] TOCTOU oversubscription test completed. Max peak observed: {:.2} MB (Budget: {} MB). Race detected: {}",
        max_mb, budget_mb, race_detected
    );

    assert!(
        !race_detected,
        "CRITICAL RACE CONDITION in MemoryBudgetGate::reserve: check and fetch_add are not atomic (TOCTOU). Allowed {:.2} MB > {} MB limit!",
        max_mb, budget_mb
    );
}


/// Stress test 2: THREAD_ARENA and bumpalo slab retention past 16 MB threshold.
#[test]
fn test_adversarial_thread_arena_slab_retention() {
    let num_threads = 16;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    let oversized_bytes = 20 * 1024 * 1024; // 20 MB > 16 MB MAX_RETAINED_BYTES

    for _ in 0..num_threads {
        let barrier_clone = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier_clone.wait();

            for _ in 0..10 {
                // Large allocation exceeding 16 MB
                with_scoped_arena(|bump| {
                    let mut vec = bumpalo::collections::Vec::with_capacity_in(oversized_bytes, bump);
                    for i in 0..oversized_bytes {
                        vec.push((i & 0xFF) as u8);
                    }
                    assert_eq!(vec.len(), oversized_bytes);
                });

                // Immediately verify the next scoped arena has been reset to baseline (<= 1MB capacity)
                with_scoped_arena(|bump| {
                    let allocated_after = bump.allocated_bytes();
                    // Bump::with_capacity(1MB) allocates ~1MB
                    assert!(
                        allocated_after <= 2 * 1024 * 1024,
                        "Oversized slab (>16MB) was NOT dropped! Retained: {} bytes",
                        allocated_after
                    );
                });
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join failed");
    }

    println!("[EMPIRICAL] THREAD_ARENA successfully dropped oversized slabs (>16 MB) across 16 threads");
}

/// Stress test 2b: Verify bumpalo chunk retention and reset semantics under repetitive 15 MB churn.
#[test]
fn test_bumpalo_reset_behavior() {
    let mut bump = bumpalo::Bump::with_capacity(1024 * 1024);
    let initial_allocated = bump.allocated_bytes();
    println!("[EMPIRICAL] Fresh Bump::with_capacity(1MB) allocated_bytes: {}", initial_allocated);

    // Allocate 15 MB (below 16 MB threshold)
    let size = 15 * 1024 * 1024;
    {
        let mut v = bumpalo::collections::Vec::with_capacity_in(size, &bump);
        for i in 0..size {
            v.push((i & 0xFF) as u8);
        }
    }
    let after_alloc = bump.allocated_bytes();
    println!("[EMPIRICAL] After 15 MB allocation, bump.allocated_bytes: {} ({:.2} MB)", after_alloc, after_alloc as f64 / (1024.0 * 1024.0));

    // Now call bump.reset()
    bump.reset();
    let after_reset = bump.allocated_bytes();
    println!("[EMPIRICAL] After bump.reset(), bump.allocated_bytes: {} ({:.2} MB)", after_reset, after_reset as f64 / (1024.0 * 1024.0));

    // Verify whether allocated_bytes in bumpalo is chunk capacity or active allocations
    assert!(
        after_alloc >= size,
        "Allocated bytes must reflect at least 15 MB"
    );
}

/// Stress test 3: High prototype churn memory bounds (< 500 MB peak RSS).
#[test]
fn test_adversarial_high_prototype_churn_process_rss() {
    let (initial_ws, initial_commit) = get_process_memory_mb();
    println!(
        "[EMPIRICAL] Process memory before churn: WS: {:.2} MB, Commit: {:.2} MB",
        initial_ws, initial_commit
    );

    let num_protos = 500;
    let num_threads = 16;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("pool build");

    let peak_ws_tracker = Arc::new(AtomicUsize::new(0));
    let stop_monitor = Arc::new(AtomicBool::new(false));

    // Background monitor thread sampling RSS every 10ms
    let monitor_tracker = Arc::clone(&peak_ws_tracker);
    let monitor_stop = Arc::clone(&stop_monitor);
    let monitor_handle = thread::spawn(move || {
        while !monitor_stop.load(Ordering::Relaxed) {
            let (ws, _) = get_process_memory_mb();
            let ws_bytes = (ws * 1024.0 * 1024.0) as usize;
            monitor_tracker.fetch_max(ws_bytes, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(10));
        }
    });

    pool.install(|| {
        use rayon::prelude::*;
        (0..num_protos).into_par_iter().for_each(|i| {
            let estimated = 256 * 1024; // 256 KB
            let _guard = MEMORY_GATE.reserve(estimated);

            with_scoped_arena(|bump| {
                // Simulate AST building & string lifting
                let mut v = bumpalo::collections::Vec::new_in(bump);
                for j in 0..10_000 {
                    v.push(j * i);
                }
                let mut s = bumpalo::collections::String::new_in(bump);
                for _ in 0..500 {
                    s.push_str("local var = 1234567890;\n");
                }
                assert!(!s.is_empty());
            });
        });
    });

    stop_monitor.store(true, Ordering::Relaxed);
    monitor_handle.join().expect("monitor join");

    let (final_ws, final_commit) = get_process_memory_mb();
    let peak_ws_mb = (peak_ws_tracker.load(Ordering::Relaxed) as f64) / (1024.0 * 1024.0);

    println!(
        "[EMPIRICAL] High churn (500 protos, 16 threads): Peak WS: {:.2} MB, Final WS: {:.2} MB, Final Commit: {:.2} MB",
        peak_ws_mb, final_ws, final_commit
    );

    assert!(
        peak_ws_mb < 500.0,
        "Peak process memory exceeded 500 MB limit! Peak WS was {:.2} MB",
        peak_ws_mb
    );
}

/// Stress test 4: 16-thread Rayon contention on MemoryBudgetGate with strict 289 MB limit.
/// Spawns 1000 tasks in a 16-thread Rayon pool, each requesting 20-60 MB to heavily oversubscribe the gate.
#[test]
fn test_adversarial_memory_gate_rayon_16_thread_contention() {
    use rayon::prelude::*;

    let budget_mb = 289;
    let budget_bytes = budget_mb * 1024 * 1024;
    let gate = Arc::new(MemoryBudgetGate::new(budget_mb));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .expect("Failed to build 16-thread Rayon pool");

    let violated = Arc::new(AtomicBool::new(false));
    let max_observed = Arc::new(AtomicUsize::new(0));
    let stop_monitor = Arc::new(AtomicBool::new(false));

    // High-frequency monitor thread sampling active_allocated_bytes
    let monitor_gate = Arc::clone(&gate);
    let monitor_violated = Arc::clone(&violated);
    let monitor_max = Arc::clone(&max_observed);
    let monitor_stop = Arc::clone(&stop_monitor);

    let monitor_handle = thread::spawn(move || {
        while !monitor_stop.load(Ordering::Acquire) {
            let active = monitor_gate.active_allocated_bytes();
            monitor_max.fetch_max(active, Ordering::Relaxed);
            if active > budget_bytes {
                monitor_violated.store(true, Ordering::SeqCst);
            }
            std::hint::spin_loop();
        }
    });

    pool.install(|| {
        (0..1000).into_par_iter().for_each(|i| {
            let gate_ref = Arc::clone(&gate);
            // Request between 20 MB and 60 MB per task
            let request_bytes = (20 + (i % 41)) * 1024 * 1024;

            let guard = gate_ref.reserve(request_bytes);
            let active = gate_ref.active_allocated_bytes();
            max_observed.fetch_max(active, Ordering::Relaxed);

            if active > budget_bytes {
                violated.store(true, Ordering::SeqCst);
            }

            // Yield briefly to increase thread interleaving and scheduling jitter
            if i % 7 == 0 {
                thread::yield_now();
            } else {
                std::hint::spin_loop();
            }

            drop(guard);
        });
    });

    stop_monitor.store(true, Ordering::Release);
    monitor_handle.join().expect("monitor join failed");

    let peak = gate.peak_allocated_bytes();
    let max_active = max_observed.load(Ordering::SeqCst);
    let peak_mb = peak as f64 / (1024.0 * 1024.0);
    let max_active_mb = max_active as f64 / (1024.0 * 1024.0);

    println!(
        "[EMPIRICAL] Rayon 16-thread contention: Peak: {:.2} MB, Max Active: {:.2} MB (Budget: {} MB, Final Active: {} bytes)",
        peak_mb,
        max_active_mb,
        budget_mb,
        gate.active_allocated_bytes()
    );

    assert_eq!(
        gate.active_allocated_bytes(),
        0,
        "All reservations must be returned to 0 after Rayon pool completes"
    );

    assert!(
        !violated.load(Ordering::SeqCst),
        "MemoryBudgetGate VIOLATED under 16-thread Rayon contention: active memory reached {:.2} MB > {} MB!",
        max_active_mb,
        budget_mb
    );

    assert!(
        peak <= budget_bytes,
        "Peak allocated bytes {:.2} MB exceeded budget limit {} MB!",
        peak_mb,
        budget_mb
    );
}

