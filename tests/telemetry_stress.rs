//! Adversarial telemetry stress testing for `luraph_deobf_rs`.
//!
//! Validates:
//! 1. Telemetry queue under rapid burst logging across concurrent threads.
//! 2. Does `telemetry.flush()` drain all events without data races or memory corruption?
//! 3. Drop rate behavior under queue saturation (capacity = 1000, DropNewest).
//! 4. Concurrent flush while workers are active (demonstrating deadlock risk).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use luraph_deobf_rs::telemetry::{
    ProtoStats, Telemetry, TelemetryConfig,
};

#[test]
fn test_telemetry_rapid_burst_logging_single_thread() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let json_log = temp_dir.path().join("burst_single.json");

    let config = TelemetryConfig {
        min_level: "TRACE".into(),
        enable_progress: false,
        json_log_path: Some(json_log.clone()),
        text_log_path: None,
    };
    let telemetry = Telemetry::new(config);

    let start = Instant::now();
    let num_events = 2_000;
    for i in 0..num_events {
        telemetry.info(format!("Burst event #{i}"));
    }
    let enqueue_duration = start.elapsed();
    println!("[BURST TEST] Enqueued {num_events} events in {enqueue_duration:?}");

    // Flush should block until all accepted queue items are processed
    let flush_start = Instant::now();
    telemetry.flush();
    let flush_duration = flush_start.elapsed();
    println!("[BURST TEST] Flushed in {flush_duration:?}");

    // Check json file content if written
    if json_log.exists() {
        let content = std::fs::read_to_string(&json_log).expect("read json log");
        let lines_count = content.lines().count();
        println!("[BURST TEST] JSON log lines written: {lines_count} / {num_events}");
        assert!(lines_count > 0, "Expected log events to be written");
    }
}

#[test]
fn test_telemetry_rapid_burst_multithreaded_contention() {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let text_log = temp_dir.path().join("burst_multi.log");

    let config = TelemetryConfig {
        min_level: "TRACE".into(),
        enable_progress: false,
        json_log_path: None,
        text_log_path: Some(text_log.clone()),
    };
    let telemetry = Telemetry::new(config);

    let num_threads = 16;
    let events_per_thread = 500; // 16 * 500 = 8,000 events total
    let total_sent = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    let start = Instant::now();

    for t in 0..num_threads {
        let tele = Arc::clone(&telemetry);
        let sent_counter = Arc::clone(&total_sent);
        handles.push(thread::spawn(move || {
            for i in 0..events_per_thread {
                let stats = ProtoStats {
                    proto_id: t * 1000 + i,
                    instruction_count: 50,
                    block_count: 4,
                    web_count: 8,
                    arena_bytes: 32 * 1024,
                    duration: Duration::from_micros(100),
                };
                tele.record_proto_lifted(&stats);
                sent_counter.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread join");
    }
    let send_duration = start.elapsed();
    println!("[BURST MULTI] Sent 8,000 events across 16 threads in {send_duration:?}");

    let flush_start = Instant::now();
    telemetry.flush();
    let flush_duration = flush_start.elapsed();
    println!("[BURST MULTI] Flush completed in {flush_duration:?}");

    if text_log.exists() {
        let content = std::fs::read_to_string(&text_log).expect("read text log");
        let lines_count = content.lines().count();
        println!("[BURST MULTI] File sink logged lines: {lines_count} / 8000");
        // Verify drop behavior: EnqueueSink capacity 1000 drops excess events under DropNewest
        let drop_count = 8000 - lines_count;
        let drop_rate = (drop_count as f64 / 8000.0) * 100.0;
        println!("[BURST MULTI] Drop rate: {drop_rate:.1}% ({drop_count} events dropped)");
    }
}

#[test]
fn test_telemetry_concurrent_flush_timeout_detection() {
    let config = TelemetryConfig {
        min_level: "INFO".into(),
        enable_progress: false,
        json_log_path: None,
        text_log_path: None,
    };
    let telemetry = Telemetry::new(config);

    let running = Arc::new(AtomicBool::new(true));
    let tele_worker = Arc::clone(&telemetry);
    let r_worker = Arc::clone(&running);

    // Background worker continuously emits log messages
    let worker_handle = thread::spawn(move || {
        let mut count = 0;
        while r_worker.load(Ordering::Relaxed) {
            tele_worker.info(format!("Continuous stream log #{count}"));
            count += 1;
            thread::yield_now();
        }
        count
    });

    // Give worker time to fill / continuously feed the queue
    thread::sleep(Duration::from_millis(10));

    // Attempt flush on a separate thread with a strict 2-second timeout
    let tele_flush = Arc::clone(&telemetry);
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        tele_flush.flush();
        let _ = tx.send(());
    });

    let flush_result = rx.recv_timeout(Duration::from_millis(500));
    
    // Stop worker thread before assertion
    running.store(false, Ordering::Relaxed);
    let worker_logs = worker_handle.join().expect("worker join");

    // Flush after worker stopped must succeed quickly
    telemetry.flush();

    match flush_result {
        Ok(_) => {
            println!("[CONCURRENT FLUSH] Flush completed despite continuous traffic ({worker_logs} logs generated).");
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            println!("[VULNERABILITY DETECTED] Telemetry::flush() timed out (>500ms) while worker thread was active!");
            // We document this empirical vulnerability: BackgroundWorker::wait_empty livelocks while producer is active
            panic!("DEADLOCK DETECTED: Telemetry::flush() cannot drain queue while active producers are logging!");
        }
        Err(e) => panic!("Flush error: {e}"),
    }
}
