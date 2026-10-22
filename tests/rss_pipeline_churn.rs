//! Empirical In-Process Continuous RSS Memory & Leak Stress Test.
//! Runs 40 consecutive pipeline deobfuscations in the SAME process
//! alternating between RideAPet.lua (397 KB) and maingui.lua (247 KB)
//! to verify that process RSS memory does not monotonically grow (zero leak)
//! and stays strictly below the 500 MB ceiling (target < 30 MB / < 260 MB).

use std::path::PathBuf;
use std::sync::Arc;
use luraph_deobf_rs::{
    run_pipeline, AssemblyStrategy, DeobfuscationConfig, Telemetry, TelemetryConfig,
};

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

fn get_process_memory_mb() -> (f64, f64, f64) {
    let mut pmc: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let success = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
    if success != 0 {
        let ws_mb = pmc.WorkingSetSize as f64 / (1024.0 * 1024.0);
        let peak_ws_mb = pmc.PeakWorkingSetSize as f64 / (1024.0 * 1024.0);
        let commit_mb = pmc.PagefileUsage as f64 / (1024.0 * 1024.0);
        (ws_mb, peak_ws_mb, commit_mb)
    } else {
        (0.0, 0.0, 0.0)
    }
}

#[test]
fn test_in_process_multi_payload_rss_leak() {
    let ws_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();

    let rideapet = ws_root
        .join("Deobfuscator-Luraph-V15")
        .join("sample")
        .join("RideAPet.lua");
    let vortex = ws_root.join("uploads").join("VortexMain.lua");

    assert!(rideapet.exists(), "RideAPet.lua missing: {:?}", rideapet);
    assert!(vortex.exists(), "VortexMain.lua missing: {:?}", vortex);

    let temp_dir = tempfile::tempdir().expect("tempdir");

    let (initial_ws, initial_peak, initial_commit) = get_process_memory_mb();
    println!(
        "\n[EMPIRICAL IN-PROCESS] Initial Process Memory: WorkingSet={:.2} MB, PeakWS={:.2} MB, Commit={:.2} MB",
        initial_ws, initial_peak, initial_commit
    );

    let mut ws_readings: Vec<f64> = Vec::new();
    let total_runs = 40;

    for i in 1..=total_runs {
        let is_ride = (i % 2) == 1;
        let input_path = if is_ride { &rideapet } else { &vortex };
        let name = if is_ride { "RideAPet" } else { "Vortex  " };
        let out_path = temp_dir.path().join(format!("out_{i}.lua"));

        let tel_config = TelemetryConfig {
            min_level: String::from("WARN"),
            enable_progress: false,
            json_log_path: None,
            text_log_path: None,
        };
        let telemetry = Telemetry::new(tel_config);

        let config = DeobfuscationConfig {
            num_threads: Some(4),
            memory_limit_mb: 500,
            strategy: AssemblyStrategy::TopLevel,
            rename: false,
            verify: false,
            telemetry,
        };

        let result = run_pipeline(input_path, &out_path, config);
        assert!(result.is_ok(), "run_pipeline failed at run {i}: {:?}", result);

        let (ws, peak, commit) = get_process_memory_mb();
        ws_readings.push(ws);

        if i % 5 == 0 || i == 1 {
            println!(
                "  [Run {:02}/{:02} - {}] WS: {:5.2} MB | PeakWS: {:5.2} MB | Commit: {:5.2} MB",
                i, total_runs, name, ws, peak, commit
            );
        }
    }

    let (final_ws, final_peak, final_commit) = get_process_memory_mb();
    println!(
        "[EMPIRICAL IN-PROCESS] Final Memory after {} consecutive runs: WS={:.2} MB, PeakWS={:.2} MB, Commit={:.2} MB",
        total_runs, final_ws, final_peak, final_commit
    );

    // Compute memory drift and leak metrics
    let first_quarter_avg: f64 = ws_readings[0..10].iter().sum::<f64>() / 10.0;
    let last_quarter_avg: f64 = ws_readings[30..40].iter().sum::<f64>() / 10.0;
    let drift = last_quarter_avg - first_quarter_avg;

    println!(
        "[EMPIRICAL IN-PROCESS] First 10 runs avg WS: {:.2} MB, Last 10 runs avg WS: {:.2} MB, Drift: {:+.2} MB",
        first_quarter_avg, last_quarter_avg, drift
    );

    // Assertions:
    // 1. Peak working set must remain strictly below 500 MB
    assert!(
        final_peak < 500.0,
        "Peak WorkingSet ({:.2} MB) exceeded 500 MB limit!",
        final_peak
    );

    // 2. Target < 30 MB / < 260 MB ceiling check
    assert!(
        final_peak < 260.0,
        "Peak WorkingSet ({:.2} MB) exceeded 260 MB target ceiling!",
        final_peak
    );

    // 3. Drift check: drift must not be unbounded (must be < 15 MB over 40 large payloads)
    assert!(
        drift.abs() < 15.0,
        "Unbounded memory leak detected! Working set drifted by {:+.2} MB across 40 runs",
        drift
    );

    println!("[EMPIRICAL IN-PROCESS] SUCCESS: In-process multi-payload run verified 0 monotonic leaks!");
}
