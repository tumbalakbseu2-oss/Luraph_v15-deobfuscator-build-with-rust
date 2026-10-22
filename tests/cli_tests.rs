use std::path::PathBuf;

use luraph_deobf_rs::{
    configure_thread_pool, run_pipeline, AssemblyStrategy, DeobfuscationConfig, DeobfError,
    Telemetry, TelemetryConfig,
};

#[test]
fn test_cli_deobf_error_exit_codes() {
    let io_err = DeobfError::Io {
        path: PathBuf::from("nonexistent.lua"),
        source: std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
    };
    assert_eq!(io_err.exit_code(), 2);

    let malformed_err = DeobfError::MalformedWire("bad header".into());
    assert_eq!(malformed_err.exit_code(), 1);

    let pipeline_err = DeobfError::Pipeline("internal error".into());
    assert_eq!(pipeline_err.exit_code(), 1);

    let verif_err = DeobfError::Verification("syntax error".into());
    assert_eq!(verif_err.exit_code(), 1);

    let oom_err = DeobfError::OutOfMemory {
        limit_mb: 500,
        active_mb: 512.0,
    };
    assert_eq!(oom_err.exit_code(), 1);

    let config_err = DeobfError::Config("invalid thread count".into());
    assert_eq!(config_err.exit_code(), 1);
}

#[test]
fn test_cli_threadpool_bounds() {
    // 0 threads must be rejected
    let res = configure_thread_pool(Some(0));
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.exit_code(), 1);

    // Valid thread count succeeds
    let valid = configure_thread_pool(Some(4));
    assert!(valid.is_ok());
}

#[test]
fn test_pipeline_nonexistent_input_returns_io_error() {
    let config = DeobfuscationConfig {
        num_threads: Some(1),
        memory_limit_mb: 500,
        strategy: AssemblyStrategy::TopLevel,
        rename: false,
        verify: false,
        telemetry: Telemetry::new(TelemetryConfig {
            min_level: "ERROR".into(),
            enable_progress: false,
            json_log_path: None,
            text_log_path: None,
        }),
    };

    let res = run_pipeline(
        &PathBuf::from("this_file_definitely_does_not_exist_12345.lua"),
        &PathBuf::from("out.lua"),
        config,
    );

    assert!(res.is_err());
    let err = res.unwrap_err();
    assert_eq!(err.exit_code(), 2, "Nonexistent file must return exit code 2");
}

#[test]
fn test_pipeline_plain_lua_rejected_exit_code_1() {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let in_path = temp_dir.path().join("input.lua");
    let out_path = temp_dir.path().join("output.luau");

    std::fs::write(&in_path, "local x = 10\nreturn x + 5\n").expect("write input");

    let config = DeobfuscationConfig {
        num_threads: Some(2),
        memory_limit_mb: 500,
        strategy: AssemblyStrategy::TopLevel,
        rename: false,
        verify: false,
        telemetry: Telemetry::new(TelemetryConfig {
            min_level: "ERROR".into(),
            enable_progress: false,
            json_log_path: None,
            text_log_path: None,
        }),
    };

    let res = run_pipeline(&in_path, &out_path, config);
    assert!(res.is_err(), "Pipeline must reject plain non-Luraph Lua script");
    let err = res.unwrap_err();
    assert_eq!(err.exit_code(), 1, "Non-Luraph input must return exit code 1");
}

#[test]
fn test_cli_args_parsing_rename_and_threads() {
    use clap::Parser;

    // Simulate CLI arguments with --rename, --strategy inline, -t 16
    #[derive(Parser, Debug)]
    struct MockCli {
        #[arg(short = 'i', long = "input")]
        input: PathBuf,
        #[arg(short = 'o', long = "output")]
        output: PathBuf,
        #[arg(short = 't', long = "threads")]
        threads: Option<usize>,
        #[arg(long = "rename", default_value_t = false)]
        rename: bool,
        #[arg(long = "strategy", default_value = "top-level")]
        strategy: String,
    }

    let parsed = MockCli::try_parse_from([
        "luraph_deobf_rs",
        "-i", "input.lua",
        "-o", "output.luau",
        "-t", "16",
        "--rename",
        "--strategy", "inline",
    ]);

    assert!(parsed.is_ok());
    let cli = parsed.unwrap();
    assert_eq!(cli.threads, Some(16));
    assert!(cli.rename);
    assert_eq!(cli.strategy, "inline");
}

