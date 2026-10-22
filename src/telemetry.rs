//! High-throughput structured execution telemetry for `luraph_deobf_rs`.
//!
//! Provides non-blocking execution logging, progress bar synchronization,
//! and structured event dispatch using `logly` and `indicatif`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

// Local logly subcrates
use logly_core::LoggerEngine;
use logly_filter::LevelFilter;
use logly_format::{JsonFormatter, TemplateFormatter};
use logly_levels::level;
use logly_record::LogRecord;
use logly_sink::{CallbackSink, EnqueueSink, FileSink};

/// Default queue capacity for asynchronous telemetry sinks (absorbs multi-threaded prototype bursts).
pub const TELEMETRY_QUEUE_CAPACITY: usize = 65_536;

/// Pipeline execution stages in the Luraph v15 deobfuscator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStage {
    /// Stage 1: Deserializing biased LEB128, opcode polynomial unhashing, prototype extraction.
    BytecodeDeserialization = 1,
    /// Stage 2: Path2D cubic Bézier curves, RK4 arc length, LPH_ENCSTR decryption.
    MathAndBezierSolving = 2,
    /// Stage 3: SCCP symbolic execution, basic block construction, control-flow unflattening.
    CfgReconstructionAndSccp = 3,
    /// Stage 4: SSA Def-Use webs, 200-local budget limiter, Roblox semantic namer.
    SsaWebbingAndNamer = 4,
    /// Stage 5: AST idiom recovery, syntax formatting, Luau source code emission.
    AstStructuringAndLuauEmission = 5,
}

impl PipelineStage {
    /// Returns the human-readable display name for the pipeline stage.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::BytecodeDeserialization => "Bytecode Deserialization & Opcode Unhashing",
            Self::MathAndBezierSolving => "Path2D Bézier Solving & Constant Decryption",
            Self::CfgReconstructionAndSccp => "CFG Reconstruction & SCCP Symbolic Execution",
            Self::SsaWebbingAndNamer => "SSA Variable Web Coalescence & Semantic Namer",
            Self::AstStructuringAndLuauEmission => "AST Structuring & Luau Source Emission",
        }
    }

    /// Returns the numeric index (1..=5) of the stage.
    #[must_use]
    pub const fn index(&self) -> usize {
        *self as usize
    }

    // Aliases for backward compatibility
    #[allow(non_upper_case_globals)]
    pub const BytecodeUnpack: Self = Self::BytecodeDeserialization;
    #[allow(non_upper_case_globals)]
    pub const CfgLift: Self = Self::CfgReconstructionAndSccp;
    #[allow(non_upper_case_globals)]
    pub const SsaWebs: Self = Self::SsaWebbingAndNamer;
    #[allow(non_upper_case_globals)]
    pub const Naming: Self = Self::SsaWebbingAndNamer;
    #[allow(non_upper_case_globals)]
    pub const LuauEmission: Self = Self::AstStructuringAndLuauEmission;
}

/// Prototype lifting telemetry statistics.
#[derive(Clone, Debug)]
pub struct ProtoStats {
    /// Unique prototype identifier.
    pub proto_id: usize,
    /// Total instruction count in this prototype.
    pub instruction_count: usize,
    /// Total basic blocks generated.
    pub block_count: usize,
    /// Total coalesced SSA variable webs.
    pub web_count: usize,
    /// Bytes allocated in the prototype-scoped bump arena.
    pub arena_bytes: usize,
    /// Time spent lifting this prototype.
    pub duration: Duration,
}

/// Final pipeline deobfuscation summary statistics.
#[derive(Clone, Debug)]
pub struct PipelineSummary {
    /// Total prototypes successfully lifted.
    pub total_protos: usize,
    /// Total instructions processed across all prototypes.
    pub total_instructions: usize,
    /// Total wall-clock time elapsed.
    pub total_duration: Duration,
    /// Peak memory consumed by arena allocations (in bytes).
    pub peak_arena_bytes: usize,
    /// Generated output size in bytes.
    pub output_bytes: usize,
}

/// Telemetry configuration options.
#[derive(Clone, Debug)]
pub struct TelemetryConfig {
    /// Minimum console logging severity level.
    pub min_level: String,
    /// Whether terminal progress bars are enabled.
    pub enable_progress: bool,
    /// Optional file path for structured JSON execution logs.
    pub json_log_path: Option<PathBuf>,
    /// Optional file path for human-readable plain text logs.
    pub text_log_path: Option<PathBuf>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            min_level: String::from("INFO"),
            enable_progress: true,
            json_log_path: None,
            text_log_path: None,
        }
    }
}

/// The centralized telemetry and logging orchestrator.
///
/// Thread-safe (`Send + Sync`) and designed to be shared across all Rayon
/// worker threads during parallel prototype lifting.
pub struct Telemetry {
    engine: RwLock<LoggerEngine>,
    multi_progress: MultiProgress,
    stage_bar: ProgressBar,
    proto_bar: ProgressBar,
    start_time: Instant,
    active_stage: RwLock<Option<PipelineStage>>,
    peak_arena_bytes: AtomicUsize,
    total_protos_lifted: AtomicUsize,
    config: TelemetryConfig,
}

impl Telemetry {
    /// Initializes the telemetry subsystem with the provided configuration.
    #[must_use]
    pub fn new(config: TelemetryConfig) -> Arc<Self> {
        let mut engine = LoggerEngine::new();
        let multi_progress = MultiProgress::new();

        if !config.enable_progress {
            multi_progress.set_draw_target(ProgressDrawTarget::hidden());
        }

        // Configure Stage Spinner
        let stage_bar = multi_progress.add(ProgressBar::new_spinner());
        if let Ok(tpl) = ProgressStyle::default_spinner()
            .template("{spinner:.cyan.bold} [{elapsed_precise}] {msg}")
        {
            stage_bar.set_style(tpl);
        }

        // Configure Proto Progress Bar
        let proto_bar = multi_progress.add(ProgressBar::new(0));
        if let Ok(tpl) = ProgressStyle::default_bar()
            .template("⚙ [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} protos ({percent}%) {msg}")
        {
            proto_bar.set_style(tpl.progress_chars("#>-"));
        }
        proto_bar.finish_and_clear();

        // 1. Console Terminal Sink (via EnqueueSink + CallbackSink)
        let mp_clone = multi_progress.clone();
        let callback = Box::new(move |line: &str| {
            // indicatif::MultiProgress::println atomically clears progress bars,
            // prints the log line, and restores the progress bar display.
            let _ = mp_clone.println(line);
        });

        let console_formatter = Box::new(TemplateFormatter::new(
            "[{time:%H:%M:%S}] [{level:<7}] {message}",
        ));
        let console_level = level(&config.min_level).unwrap_or_else(|_| level("INFO").unwrap());
        let console_filter = Box::new(LevelFilter::new(console_level));

        let callback_sink = Arc::new(CallbackSink::new(
            callback,
            console_formatter,
            console_filter,
        ));

        // Wrap callback sink in EnqueueSink for non-blocking caller dispatch
        let async_console_sink = Arc::new(EnqueueSink::with_capacity(callback_sink, TELEMETRY_QUEUE_CAPACITY));
        engine.add_sink(async_console_sink);

        // 2. Optional Structured JSON Log File Sink
        if let Some(ref path) = config.json_log_path {
            if let Ok(file_sink) = FileSink::open(
                path,
                Box::new(JsonFormatter::new()),
                Box::new(LevelFilter::new(level("TRACE").unwrap())),
                true,
                logly_rotate::RotationPolicy::Never,
                None,
                logly_config::RetentionPolicy::default(),
                logly_compress::CompressionCodec::None,
                false,
                false,
            ) {
                let async_file_sink = Arc::new(EnqueueSink::with_capacity(Arc::new(file_sink), TELEMETRY_QUEUE_CAPACITY));
                engine.add_sink(async_file_sink);
            }
        }

        // 3. Optional Plain Text Log File Sink
        if let Some(ref path) = config.text_log_path {
            if let Ok(file_sink) = FileSink::open(
                path,
                Box::new(TemplateFormatter::new(
                    "[{time:%Y-%m-%d %H:%M:%S%.3f}] [{level:<7}] [{name}] {message}",
                )),
                Box::new(LevelFilter::new(level("DEBUG").unwrap())),
                true,
                logly_rotate::RotationPolicy::Never,
                None,
                logly_config::RetentionPolicy::default(),
                logly_compress::CompressionCodec::None,
                false,
                false,
            ) {
                let async_file_sink = Arc::new(EnqueueSink::with_capacity(Arc::new(file_sink), TELEMETRY_QUEUE_CAPACITY));
                engine.add_sink(async_file_sink);
            }
        }

        Arc::new(Self {
            engine: RwLock::new(engine),
            multi_progress,
            stage_bar,
            proto_bar,
            start_time: Instant::now(),
            active_stage: RwLock::new(None),
            peak_arena_bytes: AtomicUsize::new(0),
            total_protos_lifted: AtomicUsize::new(0),
            config,
        })
    }

    /// Returns a reference to the active MultiProgress instance.
    #[must_use]
    pub fn multi_progress(&self) -> &MultiProgress {
        &self.multi_progress
    }

    /// Returns a reference to the active TelemetryConfig.
    #[must_use]
    pub fn config(&self) -> &TelemetryConfig {
        &self.config
    }

    /// Dispatches a pre-built LogRecord to all configured sinks.
    pub fn dispatch(&self, record: &LogRecord) {
        if let Ok(engine) = self.engine.read() {
            let _ = engine.dispatch(record);
        }
    }

    /// Emits a standard log message at the specified level string.
    pub fn log(&self, level_name: &str, message: &str) {
        let Ok(lvl) = level(level_name) else { return };
        let record = LogRecord::builder(lvl, message)
            .name("luraph_deobf")
            .build();
        self.dispatch(&record);
    }

    /// Emits an INFO level message.
    pub fn info(&self, message: impl AsRef<str>) {
        self.log("INFO", message.as_ref());
    }

    /// Emits a DEBUG level message.
    pub fn debug(&self, message: impl AsRef<str>) {
        self.log("DEBUG", message.as_ref());
    }

    /// Emits a WARNING level message.
    pub fn warn(&self, message: impl AsRef<str>) {
        self.log("WARNING", message.as_ref());
    }

    /// Emits an ERROR level message.
    pub fn error(&self, message: impl AsRef<str>) {
        self.log("ERROR", message.as_ref());
    }

    /// Emits a SUCCESS level message.
    pub fn success(&self, message: impl AsRef<str>) {
        self.log("SUCCESS", message.as_ref());
    }

    /// Starts a major pipeline stage.
    pub fn start_stage(&self, stage: PipelineStage, total_items: Option<usize>) {
        if let Ok(mut active) = self.active_stage.write() {
            *active = Some(stage);
        }

        let stage_text = format!(
            "Stage {}/5: {}",
            style(stage.index()).bold().cyan(),
            style(stage.name()).bold().white()
        );
        self.stage_bar.set_message(stage_text);
        self.stage_bar.enable_steady_tick(Duration::from_millis(80));

        let mut builder = LogRecord::builder(
            level("INFO").unwrap(),
            format!("Starting Stage {}/5: {}", stage.index(), stage.name()),
        )
        .name("pipeline::stage")
        .extra("event", "stage_start")
        .extra("stage", stage.name())
        .extra("stage_idx", stage.index().to_string());

        if let Some(total) = total_items {
            builder = builder.extra("total_items", total.to_string());
            if stage == PipelineStage::CfgReconstructionAndSccp {
                self.proto_bar.set_length(total as u64);
                self.proto_bar.set_position(0);
                self.proto_bar.reset();
            }
        }

        self.dispatch(&builder.build());
    }

    /// Completes a major pipeline stage.
    pub fn complete_stage(&self, stage: PipelineStage, elapsed: Duration, items_processed: usize) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        let msg = format!(
            "Completed Stage {}/5: {} in {:.2}ms ({} items)",
            stage.index(),
            stage.name(),
            elapsed_ms,
            items_processed
        );

        let record = LogRecord::builder(level("SUCCESS").unwrap(), msg)
            .name("pipeline::stage")
            .extra("event", "stage_complete")
            .extra("stage", stage.name())
            .extra("stage_idx", stage.index().to_string())
            .extra("elapsed_ms", format!("{elapsed_ms:.2}"))
            .extra("items_processed", items_processed.to_string())
            .build();

        self.dispatch(&record);
        self.flush();
    }

    /// Records metrics for a single lifted prototype.
    ///
    /// Called concurrently from multi-threaded Rayon worker tasks.
    pub fn record_proto_lifted(&self, stats: &ProtoStats) {
        self.total_protos_lifted.fetch_add(1, Ordering::Relaxed);
        let arena_kb = stats.arena_bytes / 1024;
        let elapsed_ms = stats.duration.as_secs_f64() * 1000.0;

        // Update atomic peak memory tracker
        let mut current_peak = self.peak_arena_bytes.load(Ordering::Relaxed);
        while stats.arena_bytes > current_peak {
            match self.peak_arena_bytes.compare_exchange_weak(
                current_peak,
                stats.arena_bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_peak = actual,
            }
        }

        // Advance progress bar
        self.proto_bar.inc(1);
        self.proto_bar.set_message(format!(
            "Proto #{}: {} instrs, {} KB arena",
            stats.proto_id, stats.instruction_count, arena_kb
        ));

        // Emit structured telemetry event
        let record = LogRecord::builder(
            level("DEBUG").unwrap(),
            format!(
                "Lifted Proto #{}: {} instrs, {} blocks, {} KB arena ({:.2}ms)",
                stats.proto_id, stats.instruction_count, stats.block_count, arena_kb, elapsed_ms
            ),
        )
        .name("pipeline::lifter")
        .extra("event", "proto_lifted")
        .extra("proto_id", stats.proto_id.to_string())
        .extra("instructions", stats.instruction_count.to_string())
        .extra("blocks", stats.block_count.to_string())
        .extra("webs", stats.web_count.to_string())
        .extra("arena_bytes", stats.arena_bytes.to_string())
        .extra("elapsed_ms", format!("{elapsed_ms:.2}"))
        .build();

        self.dispatch(&record);
    }

    /// Records instantaneous memory arena usage.
    pub fn record_arena_usage(&self, stage: PipelineStage, allocated_bytes: usize) {
        let arena_mb = (allocated_bytes as f64) / (1024.0 * 1024.0);
        let record = LogRecord::builder(
            level("TRACE").unwrap(),
            format!("Arena usage [{}]: {:.2} MB", stage.name(), arena_mb),
        )
        .name("memory::arena")
        .extra("event", "arena_stats")
        .extra("stage", stage.name())
        .extra("allocated_bytes", allocated_bytes.to_string())
        .build();

        self.dispatch(&record);
    }

    /// Flushes all background worker queues and attached sink destinations.
    pub fn flush(&self) {
        if let Ok(engine) = self.engine.read() {
            let _ = engine.complete();
        }
    }

    /// Emits the final execution summary and finishes progress displays.
    pub fn finish(&self, summary: &PipelineSummary) {
        self.stage_bar.finish_with_message("Deobfuscation Completed");
        self.proto_bar.finish_and_clear();

        let total_ms = summary.total_duration.as_secs_f64() * 1000.0;
        let peak_mb = (summary.peak_arena_bytes as f64) / (1024.0 * 1024.0);

        let summary_msg = format!(
            "Deobfuscation finished: {} protos, {} instructions in {:.2}ms (Peak Memory: {:.2} MB, Output: {} bytes)",
            summary.total_protos,
            summary.total_instructions,
            total_ms,
            peak_mb,
            summary.output_bytes
        );

        let record = LogRecord::builder(level("SUCCESS").unwrap(), summary_msg)
            .name("pipeline::summary")
            .extra("event", "pipeline_summary")
            .extra("total_protos", summary.total_protos.to_string())
            .extra("total_instructions", summary.total_instructions.to_string())
            .extra("total_elapsed_ms", format!("{total_ms:.2}"))
            .extra("peak_memory_mb", format!("{peak_mb:.2}"))
            .extra("output_bytes", summary.output_bytes.to_string())
            .build();

        self.dispatch(&record);
        self.flush();
    }

    /// Returns elapsed time since telemetry initialization.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Convenience macros for ergonomic logging throughout the codebase.
#[macro_export]
macro_rules! tele_info {
    ($tele:expr, $($arg:tt)*) => {
        $tele.info(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! tele_debug {
    ($tele:expr, $($arg:tt)*) => {
        $tele.debug(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! tele_warn {
    ($tele:expr, $($arg:tt)*) => {
        $tele.warn(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! tele_error {
    ($tele:expr, $($arg:tt)*) => {
        $tele.error(format!($($arg)*))
    };
}

#[macro_export]
macro_rules! tele_success {
    ($tele:expr, $($arg:tt)*) => {
        $tele.success(format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_lifecycle() {
        let config = TelemetryConfig {
            min_level: "DEBUG".into(),
            enable_progress: false,
            json_log_path: None,
            text_log_path: None,
        };
        let telemetry = Telemetry::new(config);

        telemetry.info("Testing info telemetry");
        telemetry.start_stage(PipelineStage::BytecodeUnpack, Some(10));
        telemetry.record_arena_usage(PipelineStage::BytecodeUnpack, 1024 * 1024);
        telemetry.complete_stage(PipelineStage::BytecodeUnpack, Duration::from_millis(50), 10);

        let proto_stats = ProtoStats {
            proto_id: 1,
            instruction_count: 120,
            block_count: 8,
            web_count: 14,
            arena_bytes: 48 * 1024,
            duration: Duration::from_millis(12),
        };
        telemetry.record_proto_lifted(&proto_stats);

        let summary = PipelineSummary {
            total_protos: 1,
            total_instructions: 120,
            total_duration: Duration::from_millis(65),
            peak_arena_bytes: 48 * 1024,
            output_bytes: 1500,
        };
        telemetry.finish(&summary);
    }
}
