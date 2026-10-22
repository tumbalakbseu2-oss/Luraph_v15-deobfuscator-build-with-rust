//! Standalone CLI runner for `luraph_deobf_rs`.
//!
//! Provides command-line argument parsing, logging initialization, and pipeline invocation.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, ValueEnum};

use luraph_deobf_rs::{
    run_pipeline, AssemblyStrategy as LibAssemblyStrategy, DeobfuscationConfig, Telemetry,
    TelemetryConfig,
};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CliAssemblyStrategy {
    /// Inline child closures directly into parent scopes.
    Inline,
    /// Emit recovered prototypes as top-level functions (safest for Luau 200-local budget).
    #[default]
    TopLevel,
    /// Clean GitHub publication mode with unified lexical inlining and presentation banner.
    Clean,
}

impl From<CliAssemblyStrategy> for LibAssemblyStrategy {
    fn from(strat: CliAssemblyStrategy) -> Self {
        match strat {
            CliAssemblyStrategy::Inline => LibAssemblyStrategy::Inline,
            CliAssemblyStrategy::TopLevel => LibAssemblyStrategy::TopLevel,
            CliAssemblyStrategy::Clean => LibAssemblyStrategy::Clean,
        }
    }
}

fn validate_threads(val: &str) -> Result<usize, String> {
    let n: usize = val
        .parse()
        .map_err(|_| format!("'{val}' is not a valid number"))?;
    if n == 0 {
        Err("thread count must be at least 1".to_string())
    } else {
        Ok(n)
    }
}

fn validate_memory_limit(val: &str) -> Result<usize, String> {
    let n: usize = val
        .parse()
        .map_err(|_| format!("'{val}' is not a valid number"))?;
    if n == 0 {
        Err("memory limit must be greater than 0 MB".to_string())
    } else {
        Ok(n)
    }
}

/// Command-line arguments for the Luraph v15 deobfuscator.
#[derive(Parser, Debug)]
#[command(
    name = "luraph_deobf_rs",
    author = "ICEDRAGON Engine",
    version = "0.1.0",
    about = "High-Performance Multi-Threaded Luraph v15 Deobfuscator & CFG Lifter in Rust",
    long_about = "Deobfuscates and lifts arbitrary Luraph v15/v14 payloads into clean, idiomatic Luau source code.\n\
                  Utilizes bumpalo scoped arenas (< 289 MB RAM) and rayon work-stealing parallelism."
)]
pub struct CliArgs {
    /// Path to the input obfuscated Luau script
    #[arg(short = 'i', long = "input", value_name = "INPUT_PATH", required = true)]
    pub input: PathBuf,

    /// Path to write the deobfuscated Luau output script
    #[arg(short = 'o', long = "output", value_name = "OUTPUT_PATH", required = true)]
    pub output: PathBuf,

    /// Number of worker threads for Rayon pool (defaults to logical core count)
    #[arg(short = 't', long = "threads", value_name = "NUM_THREADS", value_parser = validate_threads)]
    pub threads: Option<usize>,

    /// Maximum memory ceiling in megabytes (hard limit enforced by memory gate)
    #[arg(short = 'm', long = "memory-limit", value_name = "MB", default_value_t = 500, value_parser = validate_memory_limit)]
    pub memory_limit_mb: usize,

    /// Prototype assembly strategy
    #[arg(long = "strategy", value_enum, default_value_t = CliAssemblyStrategy::TopLevel)]
    pub strategy: CliAssemblyStrategy,

    /// Format output for publication-ready GitHub release.
    ///
    /// Automatically applies:
    /// - Professional release header banner with target metadata and engine specs
    /// - Automatic lexical inlining and copy-propagation
    /// - Dead prototype, empty stub, and unreferenced closure pruning
    /// - Semantic variable renaming and Roblox API identifier inference
    /// - Upvalue proxy reduction and lexical binding
    /// - Elimination of raw decompilation comments (-- Shared proto #X)
    /// - Idiomatic Luau function formatting
    #[arg(long = "clean", alias = "pretty", default_value_t = false)]
    pub clean: bool,

    /// Enable semantic variable renaming and Roblox API identifier inference
    #[arg(long = "rename", default_value_t = false)]
    pub rename: bool,

    /// Run post-deobfuscation syntax verification with lune/luau
    #[arg(long = "verify", default_value_t = false)]
    pub verify: bool,

    /// Disable terminal progress bar and spinner displays
    #[arg(long = "no-progress", default_value_t = false)]
    pub no_progress: bool,

    /// Optional file path for structured JSON execution logs
    #[arg(long = "json-log", value_name = "PATH")]
    pub json_log: Option<PathBuf>,

    /// Optional file path for human-readable plain text logs
    #[arg(long = "text-log", value_name = "PATH")]
    pub text_log: Option<PathBuf>,

    /// Verbosity level (-v: INFO, -vv: DEBUG, -vvv: TRACE)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbosity: u8,
}

fn main() -> ExitCode {
    let args = CliArgs::parse();

    // Map verbosity count to severity string
    let min_level = match args.verbosity {
        0 => "INFO",
        1 => "INFO",
        2 => "DEBUG",
        _ => "TRACE",
    };

    let telemetry_config = TelemetryConfig {
        min_level: min_level.to_string(),
        enable_progress: !args.no_progress,
        json_log_path: args.json_log,
        text_log_path: args.text_log,
    };

    let telemetry = Telemetry::new(telemetry_config);
    telemetry.info(format!(
        "Luraph v15 Deobfuscator (ICEDRAGON Engine) starting on {}",
        args.input.display()
    ));

    let clean_mode = args.clean;
    let strategy = if clean_mode {
        LibAssemblyStrategy::Clean
    } else {
        args.strategy.into()
    };

    let config = DeobfuscationConfig {
        num_threads: args.threads.or(Some(10)),
        memory_limit_mb: args.memory_limit_mb,
        strategy,
        verify: args.verify,
        telemetry: Arc::clone(&telemetry),
        rename: args.rename || clean_mode,
    };

    match run_pipeline(&args.input, &args.output, config) {
        Ok(summary) => {
            telemetry.success(format!(
                "Successfully generated deobfuscated payload: {} ({} bytes)",
                args.output.display(),
                summary.output_bytes
            ));
            telemetry.flush();
            ExitCode::SUCCESS
        }
        Err(err) => {
            let code = err.exit_code();
            eprintln!("error: Deobfuscation failed (exit code {code}): {err}");
            telemetry.error(format!("Deobfuscation failed (exit code {code}): {err}"));
            telemetry.flush();
            ExitCode::from(code)
        }
    }
}
