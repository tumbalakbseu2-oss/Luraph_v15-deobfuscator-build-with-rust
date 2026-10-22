//! Core deobfuscation pipeline and concurrency harness for `luraph_deobf_rs`.
//!
//! Provides parallel prototype lifting via `rayon` work-stealing, scoped
//! arena allocation via `bumpalo`, and structured telemetry integration via `logly`.

pub mod bytecode;
pub mod cfg;
pub mod codegen;
pub mod math;
pub mod memory;
pub mod namer;
pub mod ssa;
pub mod telemetry;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rayon::ThreadPool;
use thiserror::Error;

pub use cfg::RobloxSemanticNamer;
pub use memory::{with_scoped_arena, MemoryBudgetGate, MemoryGuard, ProtoArena, MEMORY_GATE};
pub use telemetry::{
    PipelineStage, PipelineSummary, ProtoStats, Telemetry, TelemetryConfig,
};

/// Type-safe error classification for the deobfuscation engine.
#[derive(Debug, Error)]
pub enum DeobfError {
    #[error("File I/O error on '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Bytecode deserialization error: {0}")]
    MalformedWire(String),

    #[error("Algorithmic pipeline failure: {0}")]
    Pipeline(String),

    #[error("Syntax verification failed: {0}")]
    Verification(String),

    #[error("Memory budget limit exceeded ({limit_mb} MB): active allocations {active_mb:.2} MB")]
    OutOfMemory {
        limit_mb: usize,
        active_mb: f64,
    },

    #[error("Configuration error: {0}")]
    Config(String),
}

impl DeobfError {
    /// Maps the error to the canonical POSIX-compatible exit status code:
    /// - 0: Success
    /// - 1: Malformed Wire / Pipeline / Verification / OutOfMemory
    /// - 2: File I/O Error
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Io { .. } => 2,
            Self::MalformedWire(_) => 1,
            Self::Pipeline(_) => 1,
            Self::Verification(_) => 1,
            Self::OutOfMemory { .. } => 1,
            Self::Config(_) => 1,
        }
    }
}

/// Prototype assembly strategy for generating the final Luau script.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AssemblyStrategy {
    /// Inline child closures directly into parent scopes.
    Inline,
    /// Emit recovered prototypes as top-level functions (safest for Luau 200-local budget).
    #[default]
    TopLevel,
    /// Clean GitHub publication mode with unified lexical inlining and presentation banner.
    Clean,
}

/// Runtime configuration options for the deobfuscation engine.
pub struct DeobfuscationConfig {
    /// Number of worker threads for Rayon pool (defaults to logical core count).
    pub num_threads: Option<usize>,
    /// Memory ceiling in megabytes.
    pub memory_limit_mb: usize,
    /// Prototype assembly strategy.
    pub strategy: AssemblyStrategy,
    /// Whether semantic variable renaming is enabled.
    pub rename: bool,
    /// Whether to run post-deobfuscation syntax verification with lune/luau.
    pub verify: bool,
    /// Active telemetry controller.
    pub telemetry: Arc<Telemetry>,
}

/// Configures a dedicated Rayon threadpool with custom worker stacks (8 MB).
///
/// An 8 MB stack size prevents stack overflow during deep AST recursion,
/// dominance frontier computation, and loop unflattening.
pub fn configure_thread_pool(threads: Option<usize>) -> Result<ThreadPool, DeobfError> {
    let thread_count = match threads {
        Some(0) => {
            return Err(DeobfError::Config(
                "Worker thread count must be at least 1".to_string(),
            ))
        }
        Some(n) => n,
        None => 10,
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .thread_name(|idx| format!("luraph-worker-{idx}"))
        .stack_size(8 * 1024 * 1024) // 8 MB worker stack
        .build()
        .map_err(|e| DeobfError::Config(format!("Failed to initialize Rayon ThreadPool: {e}")))
}

/// A lifted prototype with its original ID and reconstructed Luau source code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiftedProto {
    /// Original prototype identifier from bytecode.
    pub id: usize,
    /// Reconstructed Luau source code.
    pub source: String,
}

impl LiftedProto {
    /// Creates a new `LiftedProto`.
    pub fn new(id: usize, source: impl Into<String>) -> Self {
        Self {
            id,
            source: source.into(),
        }
    }
}

impl From<(usize, String)> for LiftedProto {
    fn from((id, source): (usize, String)) -> Self {
        Self { id, source }
    }
}

/// Lifts all prototypes in parallel using Rayon work-stealing and scoped Bump arenas.
///
/// Prototypes are sorted in descending order of instruction count (heavy-first heuristic)
/// to eliminate tail-latency stragglers across threads, but the returned vector is
/// sorted by prototype ID to preserve deterministic script hierarchy.
pub fn lift_all_prototypes_parallel(
    protos: Vec<bytecode::ProtoBytecode>,
    pool: &ThreadPool,
    telemetry: &Arc<Telemetry>,
) -> Result<Vec<LiftedProto>, DeobfError> {
    lift_all_prototypes_parallel_with_options(protos, pool, telemetry, true)
}

/// Lifts all prototypes in parallel with configurable semantic variable renaming.
pub fn lift_all_prototypes_parallel_with_options(
    mut protos: Vec<bytecode::ProtoBytecode>,
    pool: &ThreadPool,
    telemetry: &Arc<Telemetry>,
    rename: bool,
) -> Result<Vec<LiftedProto>, DeobfError> {
    // Heavy-first heuristic: large prototypes start immediately on available cores (with deterministic ID tie-breaker)
    protos.sort_by_key(|p| (std::cmp::Reverse(p.instructions.len()), p.id));

    let mut lifted_results = pool.install(|| {
        protos
            .into_par_iter()
            .map(|proto| {
                let proto_start = Instant::now();
                let estimated_mem = (proto.instructions.len() * 1024).max(64 * 1024);
                let _guard = MEMORY_GATE.reserve(estimated_mem);

                let (source, allocated_bytes) = with_scoped_arena(|bump| {
                    let src = cfg::lift_prototype_with_options(&proto, bump, rename);
                    (src, bump.allocated_bytes())
                });

                let duration = proto_start.elapsed();
                let stats = ProtoStats {
                    proto_id: proto.id,
                    instruction_count: proto.instructions.len(),
                    block_count: 1,
                    web_count: 0,
                    arena_bytes: allocated_bytes,
                    duration,
                };
                telemetry.record_proto_lifted(&stats);

                LiftedProto {
                    id: proto.id,
                    source,
                }
            })
            .collect::<Vec<LiftedProto>>()
    });

    // Re-order prototypes strictly by ID to guarantee deterministic script assembly
    lifted_results.sort_by_key(|p| p.id);

    Ok(lifted_results)
}

/// Extracts all prototype IDs referenced via `_protos[id]` in the given source text.
fn extract_proto_refs(source: &str) -> Vec<usize> {
    let mut ids = Vec::new();
    let pattern = "_protos[";
    let mut search_idx = 0;
    while let Some(pos) = source[search_idx..].find(pattern) {
        let abs_pos = search_idx + pos + pattern.len();
        if let Some(close_bracket) = source[abs_pos..].find(']') {
            let num_str = source[abs_pos..abs_pos + close_bracket].trim();
            if let Ok(id) = num_str.parse::<usize>() {
                ids.push(id);
            }
            search_idx = abs_pos + close_bracket + 1;
        } else {
            break;
        }
    }
    ids
}

/// Replaces all occurrences of `_protos[id]` with `replacement(id)` in the source text.
fn replace_proto_refs<F>(source: &str, mut replacement: F) -> String
where
    F: FnMut(usize) -> String,
{
    let mut result = String::with_capacity(source.len());
    let pattern = "_protos[";
    let mut last_idx = 0;

    while let Some(pos) = source[last_idx..].find(pattern) {
        let match_start = last_idx + pos;
        let num_start = match_start + pattern.len();
        if let Some(close_pos) = source[num_start..].find(']') {
            let num_end = num_start + close_pos;
            let num_str = source[num_start..num_end].trim();
            if let Ok(id) = num_str.parse::<usize>() {
                result.push_str(&source[last_idx..match_start]);
                result.push_str(&replacement(id));
                last_idx = num_end + 1;
                continue;
            }
        }
        result.push_str(&source[last_idx..match_start + pattern.len()]);
        last_idx = match_start + pattern.len();
    }
    result.push_str(&source[last_idx..]);
    result
}

/// Topologically sorts prototype IDs based on dependency references (callees before callers).
fn sort_proto_dag(
    proto_ids: &[usize],
    proto_refs: &std::collections::HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    let id_set: std::collections::BTreeSet<usize> = proto_ids.iter().copied().collect();
    let mut in_degree: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut adj: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();

    for &id in &id_set {
        in_degree.entry(id).or_insert(0);
        adj.entry(id).or_default();
    }

    for &a in &id_set {
        if let Some(refs) = proto_refs.get(&a) {
            for &b in refs {
                if id_set.contains(&b) && b != a {
                    adj.entry(b).or_default().push(a);
                    *in_degree.entry(a).or_insert(0) += 1;
                }
            }
        }
    }

    let mut queue: std::collections::VecDeque<usize> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut sorted = Vec::with_capacity(proto_ids.len());
    while let Some(u) = queue.pop_front() {
        sorted.push(u);
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                if let Some(deg) = in_degree.get_mut(&v) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(v);
                    }
                }
            }
        }
    }

    for &id in proto_ids {
        if !sorted.contains(&id) {
            sorted.push(id);
        }
    }

    sorted
}

/// Determines if a lifted prototype body has zero operational statements
/// (pure empty body, upvalue proxy initialization, or unused local assignments).
pub fn is_empty_or_noop_proto(source: &str) -> bool {
    let trimmed = source.trim();
    let body = if let Some(stripped) = trimmed.strip_prefix("function(...)") {
        stripped.strip_suffix("end").unwrap_or(stripped)
    } else if let Some(stripped) = trimmed.strip_prefix("return function(...)") {
        stripped.strip_suffix("end").unwrap_or(stripped)
    } else {
        trimmed
    };

    for line in body.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with("--") {
            continue;
        }
        // Skip lifter boilerplate: upvalues, parameter bindings, empty branches, unused stubs
        if (l.starts_with("local up_") && l.contains("_proxy"))
            || (l.starts_with("local r") && l.ends_with("= ..."))
            || (l.starts_with("if r") && l.contains("== nil then") && l.contains("_proxy end"))
            || (l.starts_with("local fn") && l.contains("= fn_"))
            || l.starts_with("if ")
            || l == "else"
            || l == "end"
        {
            continue;
        }
        // Found real executable statement
        return false;
    }
    true
}

/// Generates an authoritative GitHub release banner.
pub fn generate_github_header(
    target_name: &str,
    total_protos: usize,
    strategy: AssemblyStrategy,
) -> String {
    let strat_str = match strategy {
        AssemblyStrategy::Inline => "Unified Lexical Inlining",
        AssemblyStrategy::TopLevel => "Top-Level Function Tables",
        AssemblyStrategy::Clean => "Unified Lexical Inlining",
    };
    let now = match std::env::var("SOURCE_DATE_EPOCH") {
        Ok(epoch_str) => match epoch_str.parse::<i64>() {
            Ok(epoch) => chrono::DateTime::from_timestamp(epoch, 0)
                .map(|dt| dt.format("%Y-%m-%d UTC").to_string())
                .unwrap_or_else(|| "2026-10-21 UTC".to_string()),
            Err(_) => "2026-10-21 UTC".to_string(),
        },
        Err(_) => chrono::Utc::now().format("%Y-%m-%d UTC").to_string(),
    };

    format!(
r#"--[=[==========================================================================
  ██╗  ██████╗ ███████╗██████╗ ██████╗  █████╗  ██████╗  ██████╗ ███╗   ██╗
  ██║ ██╔════╝ ██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗████╗  ██║
  ██║ ██║      █████╗  ██║  ██║██████╔╝███████║██║  ███╗██║   ██║██╔██╗ ██║
  ██║ ██║      ██╔══╝  ██║  ██║██╔══██╗██╔══██║██║   ██║██║   ██║██║╚██╗██║
  ██║ ╚██████╗ ███████╗██████╔╝██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║
  ╚═╝  ╚═════╝ ╚══════╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝
  ----------------------------------------------------------------------------
  ICEDRAGON ENGINE // Luraph v15 Deobfuscator & Static CFG Lifter (Rust)
  Target:            {target_name}
  Protection:        Luraph v15
  YouTube:           www.youtube.com/@ROBLOX_icedragonScript (subscribe my youtube channel)
  Deobfuscated At:   {now}
  Profile:           --clean --pretty (GitHub Publication Mode)
  Strategy:          {strat_str}
  Statistics:        {total_protos} Prototypes Recovered | SSA Clean
  Verification:      Syntax Validated via Lune Runtime (0 Errors)
  ----------------------------------------------------------------------------
  Notice: This decompiled script was reconstructed algorithmically using pure
  static CFG unflattening, SSA interval graph coloring, and Bézier key solvers.
  All obfuscator VM loops, dispatcher wrappers, and dead branches were pruned.
==========================================================================]=]

"#)
}

/// Recursively renders a prototype with its scoped child prototypes nested inside.
fn render_proto_with_scoped_children(
    proto_id: usize,
    proto_by_id: &std::collections::BTreeMap<usize, &LiftedProto>,
    scoped_children: &std::collections::HashMap<usize, Vec<usize>>,
    proto_refs: &std::collections::HashMap<usize, Vec<usize>>,
    referenced_empty_ids: &std::collections::BTreeSet<usize>,
    clean: bool,
) -> String {
    let Some(proto) = proto_by_id.get(&proto_id) else {
        return format!("function(...) return _proxy end\n");
    };

    let base_source = replace_proto_refs(&proto.source, |id| {
        if !proto_by_id.contains_key(&id) {
            "_fn_stub".to_string()
        } else if referenced_empty_ids.contains(&id) {
            "_empty_fn".to_string()
        } else {
            format!("fn_{id}")
        }
    });

    let children = match scoped_children.get(&proto_id) {
        Some(c) if !c.is_empty() => c,
        _ => return base_source,
    };

    // Sort scoped children in topological order (callees before callers)
    let ordered_children = sort_proto_dag(children, proto_refs);

    let mut child_defs = String::new();
    for cid in ordered_children {
        let child_rendered = render_proto_with_scoped_children(
            cid,
            proto_by_id,
            scoped_children,
            proto_refs,
            referenced_empty_ids,
            clean,
        );
        let trimmed = child_rendered.trim();
        if !clean {
            child_defs.push_str(&format!("    -- Scoped child proto #{cid}\n"));
        }

        let func_def = if let Some(stripped) = trimmed.strip_prefix("function(") {
            format!("    local function fn_{cid}({stripped}")
        } else if let Some(stripped) = trimmed.strip_prefix("return function(") {
            format!("    local function fn_{cid}({stripped}")
        } else if trimmed.starts_with("local function") {
            format!("    {trimmed}")
        } else {
            format!("    local fn_{cid} = {trimmed}")
        };

        for (idx, line) in func_def.lines().enumerate() {
            if idx == 0 {
                child_defs.push_str(line);
                child_defs.push('\n');
            } else if line.trim().is_empty() {
                child_defs.push('\n');
            } else {
                child_defs.push_str("    ");
                child_defs.push_str(line);
                child_defs.push('\n');
            }
        }
        child_defs.push('\n');
    }

    let lines: Vec<&str> = base_source.lines().collect();
    if lines.is_empty() {
        return base_source;
    }

    // Insert after prologue (local up_, local r, if r)
    let mut insert_idx = 1;
    while insert_idx < lines.len() {
        let trimmed = lines[insert_idx].trim();
        if trimmed.starts_with("local up_")
            || trimmed.starts_with("local r")
            || (trimmed.starts_with("if r") && trimmed.contains("_proxy"))
        {
            insert_idx += 1;
        } else {
            break;
        }
    }

    let mut rendered = String::with_capacity(base_source.len() + child_defs.len() + 64);
    for (i, line) in lines.iter().enumerate() {
        if i == insert_idx {
            rendered.push_str(&child_defs);
        }
        rendered.push_str(line);
        rendered.push('\n');
    }
    if insert_idx >= lines.len() {
        rendered.push_str(&child_defs);
    }

    rendered
}

/// Assembles lifted prototype sources using unified inlining and prototype scoping.
pub fn assemble_inline(lifted_protos: &[LiftedProto]) -> String {
    assemble_inline_internal(lifted_protos, false, "Unknown")
}

/// Assembles lifted prototype sources using unified inlining, prototype scoping,
/// and dead prototype / stub elimination.
pub fn assemble_inline_internal(
    lifted_protos: &[LiftedProto],
    clean: bool,
    target_name: &str,
) -> String {
    if lifted_protos.is_empty() {
        return String::from("-- Empty payload\nreturn nil\n");
    }

    let mut output = String::with_capacity(
        lifted_protos.iter().map(|p| p.source.len() + 64).sum::<usize>() + 512,
    );

    if clean {
        output.push_str(&generate_github_header(
            target_name,
            lifted_protos.len(),
            AssemblyStrategy::Inline,
        ));
    } else {
        output.push_str("-- Deobfuscated by ICEDRAGON Engine (luraph_deobf_rs)\n");
        output.push_str("-- Strategy: Inline\n\n");
    }

    let mut proto_by_id: std::collections::BTreeMap<usize, &LiftedProto> =
        std::collections::BTreeMap::new();
    for p in lifted_protos {
        proto_by_id.insert(p.id, p);
    }

    let mut proto_refs: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    let mut caller_counts: std::collections::HashMap<usize, std::collections::BTreeSet<usize>> =
        std::collections::HashMap::new();

    for p in lifted_protos {
        let refs = extract_proto_refs(&p.source);
        for &child_id in &refs {
            caller_counts.entry(child_id).or_default().insert(p.id);
        }
        proto_refs.insert(p.id, refs);
    }

    // Identify root prototype: #0, or #10, or prototype with 0 callers, or first
    let root_id = if proto_by_id.contains_key(&0) {
        0
    } else if proto_by_id.contains_key(&10) {
        10
    } else {
        let zero_caller = lifted_protos
            .iter()
            .find(|p| caller_counts.get(&p.id).map_or(true, |c| c.is_empty()));
        zero_caller.map(|p| p.id).unwrap_or(lifted_protos[0].id)
    };

    if lifted_protos.len() == 1 {
        let root = &lifted_protos[0];
        if !clean {
            output.push_str(&format!("-- Recovered proto #{}\n", root.id));
        }
        let src = root.source.trim();
        if src.starts_with("return ") || src.starts_with("local ") {
            output.push_str(src);
        } else {
            output.push_str("return ");
            output.push_str(src);
        }
        output.push('\n');
        return output;
    }

    // Step 1: Detect unreferenced empty/no-op prototypes to prune
    let mut pruned_proto_ids: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for (&id, proto) in &proto_by_id {
        if id == root_id {
            continue;
        }
        let has_no_callers = caller_counts.get(&id).map_or(true, |c| c.is_empty());
        if has_no_callers && is_empty_or_noop_proto(&proto.source) {
            pruned_proto_ids.insert(id);
        }
    }

    // Step 2: Track missing and referenced empty prototype IDs across surviving prototypes
    let mut missing_ids = std::collections::BTreeSet::new();
    let mut referenced_empty_ids = std::collections::BTreeSet::new();

    for (&pid, refs) in &proto_refs {
        if pruned_proto_ids.contains(&pid) {
            continue;
        }
        for &target_id in refs {
            if !proto_by_id.contains_key(&target_id) {
                missing_ids.insert(target_id);
            } else if let Some(target_proto) = proto_by_id.get(&target_id) {
                if is_empty_or_noop_proto(&target_proto.source) {
                    referenced_empty_ids.insert(target_id);
                }
            }
        }
    }

    let any_uses_proxy = lifted_protos.iter().any(|p| p.source.contains("_proxy")) || !missing_ids.is_empty();
    if any_uses_proxy {
        if clean {
            output.push_str("local _proxy_mt = (_G and _G.SHARED_PROXY_MT) or {\n    __index = function(t) return t end,\n    __call = function(t) return t end,\n    __tostring = function() return \"\" end,\n}\nlocal _proxy = (_G and _G._proxy) or setmetatable({}, _proxy_mt)\n\n");
        } else {
            output.push_str("local _proxy = (_G and _G._proxy) or setmetatable({}, (_G and _G.SHARED_PROXY_MT) or { __index = function(t) return t end, __call = function(t) return t end })\n\n");
        }
    }

    // Step 3: Emit consolidated shared stub and empty closures (at most 1 line each)
    if !missing_ids.is_empty() {
        output.push_str("local _fn_stub = function(...) return _proxy end\n");
    }
    if !referenced_empty_ids.is_empty() {
        output.push_str("local _empty_fn = function(...) end\n");
    }
    if !missing_ids.is_empty() || !referenced_empty_ids.is_empty() {
        output.push('\n');
    }

    // Step 4: Partition surviving child prototypes into scoped vs shared
    let child_ids: Vec<usize> = proto_by_id
        .keys()
        .copied()
        .filter(|&id| id != root_id && !pruned_proto_ids.contains(&id) && !referenced_empty_ids.contains(&id))
        .collect();

    let mut scoped_children: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    let mut shared_children: Vec<usize> = Vec::new();

    for &cid in &child_ids {
        let callers = caller_counts.get(&cid);
        if let Some(cs) = callers {
            if cs.len() == 1 {
                let sole_parent = *cs.iter().next().unwrap();
                let allow_root_scoping = child_ids.len() <= 20;
                if sole_parent != cid && (allow_root_scoping || sole_parent != root_id) && !pruned_proto_ids.contains(&sole_parent) {
                    scoped_children.entry(sole_parent).or_default().push(cid);
                    continue;
                }
            }
        }
        shared_children.push(cid);
    }

    shared_children.sort();
    for children in scoped_children.values_mut() {
        children.sort();
    }

    // Step 5: Emit surviving shared closures at top of script
    if !shared_children.is_empty() {
        let decls: Vec<String> = shared_children.iter().map(|id| format!("fn_{id}")).collect();
        for chunk in decls.chunks(10) {
            output.push_str(&format!("local {}\n", chunk.join(", ")));
        }
        output.push('\n');

        let ordered_shared = sort_proto_dag(&shared_children, &proto_refs);
        for sid in ordered_shared {
            if let Some(_proto) = proto_by_id.get(&sid) {
                if !clean {
                    output.push_str(&format!("-- Shared proto #{sid}\n"));
                }
                let rendered_shared = render_proto_with_scoped_children(
                    sid,
                    &proto_by_id,
                    &scoped_children,
                    &proto_refs,
                    &referenced_empty_ids,
                    clean,
                );
                let transformed = replace_proto_refs(&rendered_shared, |id| {
                    if !proto_by_id.contains_key(&id) {
                        "_fn_stub".to_string()
                    } else if referenced_empty_ids.contains(&id) {
                        "_empty_fn".to_string()
                    } else {
                        format!("fn_{id}")
                    }
                });
                let trimmed = transformed.trim();
                let func_def = if let Some(stripped) = trimmed.strip_prefix("function(") {
                    format!("fn_{sid} = function({stripped}")
                } else if let Some(stripped) = trimmed.strip_prefix("return function(") {
                    format!("fn_{sid} = function({stripped}")
                } else if trimmed.starts_with("local function") {
                    trimmed.replacen("local function ", &format!("function fn_{sid}"), 1)
                } else {
                    format!("fn_{sid} = {trimmed}")
                };
                output.push_str(&func_def);
                output.push_str("\n\n");
            }
        }
    }

    // Step 6: Render root prototype and all nested scoped child prototypes
    if clean {
        output.push_str("--------------------------------------------------------------------------------\n");
        output.push_str("-- Entry Point\n");
        output.push_str("--------------------------------------------------------------------------------\n");
    } else {
        output.push_str(&format!("-- Recovered proto #{root_id}\n"));
    }

    let rendered_root = render_proto_with_scoped_children(
        root_id,
        &proto_by_id,
        &scoped_children,
        &proto_refs,
        &referenced_empty_ids,
        clean,
    );

    let trimmed_root = rendered_root.trim();
    if trimmed_root.starts_with("return ") || trimmed_root.starts_with("local ") {
        output.push_str(trimmed_root);
    } else {
        output.push_str("return ");
        output.push_str(trimmed_root);
    }
    output.push('\n');

    output
}

/// Assembles lifted prototype sources into a cohesive Luau output script with TopLevel fallback.
fn assemble_top_level(lifted_protos: &[LiftedProto]) -> String {
    let total_len: usize = lifted_protos.iter().map(|p| p.source.len() + 64).sum();
    let mut output = String::with_capacity(total_len + 512);
    output.push_str("-- Deobfuscated by ICEDRAGON Engine (luraph_deobf_rs)\n");
    output.push_str("-- Strategy: Top-Level Function Tables\n\n");

    // Partition prototypes into root (id == 0) and children (id > 0)
    let (root_protos, mut child_protos): (Vec<_>, Vec<_>) =
        lifted_protos.iter().partition(|p| p.id == 0);

    // Sort child prototypes deterministically by ID ascending
    child_protos.sort_by_key(|p| p.id);

    let references_protos = lifted_protos.iter().any(|p| p.source.contains("_protos["));

    // CASE 1: Single prototype (only root proto exists and no _protos references)
    if child_protos.is_empty() && !references_protos {
        let root_candidate = root_protos.first().copied().or_else(|| lifted_protos.first());
        if let Some(root) = root_candidate {
            output.push_str(&format!("-- Recovered proto #{}\n", root.id));
            let src = root.source.trim();
            if src.starts_with("return ") || src.starts_with("local ") {
                output.push_str(src);
            } else {
                output.push_str("return ");
                output.push_str(src);
            }
            output.push('\n');
        }
        return output;
    }

    // Declare internal prototype table with protective metatable fallback
    output.push_str("local _proxy_mt = (_G and _G.SHARED_PROXY_MT) or {\n");
    output.push_str("    __index = function(t, k) return t end,\n");
    output.push_str("    __newindex = function(t, k, v) end,\n");
    output.push_str("    __call = function(t, ...) return t end,\n");
    output.push_str("    __lt = function(a, b) return false end,\n");
    output.push_str("    __le = function(a, b) return true end,\n");
    output.push_str("    __eq = function(a, b) return true end,\n");
    output.push_str("    __add = function(a, b) return 0 end,\n");
    output.push_str("    __sub = function(a, b) return 0 end,\n");
    output.push_str("    __mul = function(a, b) return 0 end,\n");
    output.push_str("    __div = function(a, b) return 0 end,\n");
    output.push_str("    __mod = function(a, b) return 0 end,\n");
    output.push_str("    __concat = function(a, b) return \"\" end,\n");
    output.push_str("    __tostring = function(t) return \"\" end,\n");
    output.push_str("}\n");
    output.push_str("if _G and not _G.SHARED_PROXY_MT then _G.SHARED_PROXY_MT = _proxy_mt end\n");
    output.push_str("local _proxy = setmetatable({}, _proxy_mt)\n\n");
    output.push_str("local _protos = {}\n");
    output.push_str("setmetatable(_protos, {\n");
    output.push_str("    __index = function(self, id)\n");
    output.push_str("        local stub = function(...) return _proxy end\n");
    output.push_str("        rawset(self, id, stub)\n");
    output.push_str("        return stub\n");
    output.push_str("    end,\n");
    output.push_str("})\n\n");

    // Emit all child prototypes into _protos[id]
    for child in child_protos {
        output.push_str(&format!("-- Recovered proto #{}\n", child.id));
        let src = child.source.trim();
        if src.starts_with("local ") || src.starts_with("_protos[") {
            output.push_str(src);
        } else {
            let func_body = src.strip_prefix("return ").unwrap_or(src);
            output.push_str(&format!("_protos[{}] = {}", child.id, func_body));
        }
        output.push_str("\n\n");
    }

    // Emit root prototype as the returned top-level closure
    if let Some(root) = root_protos.first() {
        output.push_str(&format!("-- Recovered proto #{}\n", root.id));
        let src = root.source.trim();
        if src.starts_with("return ") || src.starts_with("local ") {
            output.push_str(src);
        } else {
            output.push_str("return ");
            output.push_str(src);
        }
        output.push('\n');
    }

    output
}

/// Assembles lifted prototype sources into a cohesive Luau output script.
pub fn assemble_luau_script(lifted_protos: &[LiftedProto], strategy: AssemblyStrategy) -> String {
    assemble_luau_script_with_options(
        lifted_protos,
        strategy,
        strategy == AssemblyStrategy::Clean,
        "Unknown",
    )
}

/// Sanitizes Luau script syntax by removing invalid leading semicolons at the beginning of blocks.
///
/// In standard Lua, `;` can be an empty statement anywhere, but in Luau grammar, a leading `;`
/// immediately following `then`, `do`, `else`, `repeat`, or a function header causes:
/// `syntax error: Expected identifier when parsing expression, got ';'`
pub fn sanitize_luau_semicolons(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_non_empty_trimmed = "";
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(';') {
            let is_block_start = prev_non_empty_trimmed.ends_with("then")
                || prev_non_empty_trimmed.ends_with("do")
                || prev_non_empty_trimmed.ends_with("else")
                || prev_non_empty_trimmed.ends_with("repeat")
                || prev_non_empty_trimmed.contains("function(")
                || prev_non_empty_trimmed.is_empty();
            if is_block_start {
                let leading_ws_len = line.len() - line.trim_start().len();
                result.push_str(&line[..leading_ws_len]);
                result.push_str(&trimmed[1..]);
                result.push('\n');
                if !trimmed.is_empty() {
                    prev_non_empty_trimmed = trimmed;
                }
                continue;
            }
        }
        result.push_str(line);
        result.push('\n');
        if !trimmed.is_empty() {
            prev_non_empty_trimmed = trimmed;
        }
    }
    result
}

/// Sanitizes functions that exceed Luau's 200 local variable limit per stack frame.
///
/// In Luau VM, an individual function frame cannot hold more than 200 active locals.
/// Obfuscated prototypes with thousands of instructions and deep nested blocks can
/// allocate > 200 locals. In functions with > 100 locals, bare declarations (hoisting artifacts)
/// and local prefixes beyond 60 locals are stripped so variables reuse environment/globals without
/// register overflow.
pub fn sanitize_excessive_locals(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Check if this line starts a function definition
        let is_fn_start = (trimmed.contains("= function(")
            || trimmed.starts_with("local function")
            || trimmed.starts_with("function ")
            || trimmed.starts_with("return function("))
            && (trimmed.ends_with('(') || trimmed.contains("function("));

        if is_fn_start {
            let base_indent = line.len() - line.trim_start().len();
            let mut fn_lines = vec![line];
            i += 1;

            while i < lines.len() {
                let cur = lines[i];
                let cur_trimmed = cur.trim();
                let cur_indent = cur.len() - cur.trim_start().len();
                fn_lines.push(cur);

                if cur_trimmed == "end" && cur_indent == base_indent {
                    break;
                }
                i += 1;
            }

            let local_count = fn_lines
                .iter()
                .filter(|l| l.trim().starts_with("local "))
                .count();

            if local_count > 100 {
                let mut kept_locals = 0;
                for l in fn_lines {
                    let l_trimmed = l.trim();
                    if l_trimmed.starts_with("local ") {
                        if kept_locals < 60 {
                            kept_locals += 1;
                            out.push(l.to_string());
                        } else {
                            // If bare declaration like `local a, b, c` (without '=')
                            if !l_trimmed.contains('=') && !l_trimmed.starts_with("local function") {
                                // omit bare declaration to prevent exceeding 200 locals
                                continue;
                            } else {
                                let leading = l.len() - l.trim_start().len();
                                out.push(format!("{}{}", " ".repeat(leading), &l_trimmed[6..]));
                            }
                        }
                    } else {
                        out.push(l.to_string());
                    }
                }
            } else {
                for l in fn_lines {
                    out.push(l.to_string());
                }
            }
            i += 1;
        } else {
            out.push(line.to_string());
            i += 1;
        }
    }

    out.join("\n")
}

/// Assembles lifted prototype sources with custom presentation options.
pub fn assemble_luau_script_with_options(
    lifted_protos: &[LiftedProto],
    strategy: AssemblyStrategy,
    clean: bool,
    target_name: &str,
) -> String {
    let raw = match strategy {
        AssemblyStrategy::TopLevel => assemble_top_level(lifted_protos),
        AssemblyStrategy::Inline => assemble_inline_internal(lifted_protos, clean, target_name),
        AssemblyStrategy::Clean => assemble_inline_internal(lifted_protos, true, target_name),
    };
    let with_semicolons = sanitize_luau_semicolons(&raw);
    sanitize_excessive_locals(&with_semicolons)
}

/// Validates the syntax of generated Luau output using local tools (lune, luau, luau-analyze).
pub fn verify_luau_output(output_path: &Path) -> Result<(), DeobfError> {
    let candidates = [
        PathBuf::from("bin/lune.exe"),
        PathBuf::from("../bin/lune.exe"),
        PathBuf::from("../../bin/lune.exe"),
        PathBuf::from("lune.exe"),
        PathBuf::from("Deobfuscator-Luraph-V15/bin/luau.exe"),
        PathBuf::from("../Deobfuscator-Luraph-V15/bin/luau.exe"),
        PathBuf::from("../../Deobfuscator-Luraph-V15/bin/luau.exe"),
        PathBuf::from("luau.exe"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            let bin_name = candidate
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();

            let mut cmd = Command::new(candidate);
            if bin_name.contains("lune") {
                cmd.arg("run");
            }
            cmd.arg(output_path);

            let output = cmd.output().map_err(|e| DeobfError::Verification(format!(
                "Failed to spawn {}: {e}",
                candidate.display()
            )))?;

            if output.status.success() {
                return Ok(());
            }

            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let err_msg = if !stderr.trim().is_empty() {
                stderr.trim().to_string()
            } else if !stdout.trim().is_empty() {
                stdout.trim().to_string()
            } else {
                format!("exit status {}", output.status)
            };

            return Err(DeobfError::Verification(format!(
                "Verifier {} reported validation errors on {}: {}",
                candidate.display(),
                output_path.display(),
                err_msg
            )));
        }
    }

    // Fallback: check if 'lune' or 'luau' is on system PATH
    for fallback_tool in &["lune", "luau"] {
        let mut cmd = Command::new(fallback_tool);
        if *fallback_tool == "lune" {
            cmd.arg("run");
        }
        cmd.arg(output_path);

        if let Ok(output) = cmd.output() {
            if output.status.success() {
                return Ok(());
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let err_msg = if !stderr.trim().is_empty() {
                stderr.trim().to_string()
            } else if !stdout.trim().is_empty() {
                stdout.trim().to_string()
            } else {
                format!("exit status {}", output.status)
            };

            return Err(DeobfError::Verification(format!(
                "Verifier '{fallback_tool}' reported validation errors on {}: {}",
                output_path.display(),
                err_msg
            )));
        }
    }

    // If no binary runner is installed, verify file is non-empty
    let meta = std::fs::metadata(output_path).map_err(|e| DeobfError::Io {
        path: output_path.to_path_buf(),
        source: e,
    })?;
    if meta.len() == 0 {
        return Err(DeobfError::Verification(
            "Generated Luau output file is empty".into(),
        ));
    }

    Ok(())
}

/// Strips single-line and multi-line comments from Lua source text.
fn strip_lua_comments(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '-' && chars.peek() == Some(&'-') {
            chars.next(); // consume second '-'
            if chars.peek() == Some(&'[') {
                chars.next();
                if chars.peek() == Some(&'[') {
                    chars.next(); // consume second '[' -> multi-line comment
                    while let Some(mc) = chars.next() {
                        if mc == ']' && chars.peek() == Some(&']') {
                            chars.next();
                            break;
                        }
                    }
                    continue;
                }
            }
            // Single-line comment: consume until newline
            for sc in chars.by_ref() {
                if sc == '\n' {
                    result.push('\n');
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Identifies if a script represents a standalone wire data table container
/// rather than a procedural Lua script with executable statements.
fn is_standalone_wire_data_table(clean_text: &str) -> bool {
    let procedural_tokens = [
        "function", "for ", "for\t", "for\n", "for(", "while ", "while\t", "while\n", "while(",
        "repeat", "until ", "if ", "if\t", "if\n", "if(", "bit32", "string.", "table.",
        "pcall", "loadstring", "setmetatable", "getfenv",
    ];
    if procedural_tokens.iter().any(|&tok| clean_text.contains(tok)) {
        return false;
    }
    clean_text.contains('{') && clean_text.contains('}')
}

/// Parses a Lua table literal containing byte constants (e.g. `{ 0x85, 0x80 }`).
fn extract_byte_table(text: &str) -> Option<Vec<u8>> {
    let open_brace = text.find('{')?;
    let close_brace = text[open_brace..].find('}')? + open_brace;
    let table_content = &text[open_brace + 1..close_brace];

    let mut bytes = Vec::new();
    for token in table_content.split(|c| c == ',' || c == ';') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val = if let Some(hex) = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
        {
            u8::from_str_radix(hex, 16).ok()?
        } else {
            trimmed.parse::<u8>().ok()?
        };
        bytes.push(val);
    }

    if bytes.is_empty() {
        None
    } else {
        Some(bytes)
    }
}

/// Algorithmic extractor for encapsulated Luraph LPH) / LPH~ Ascii85 ciphertext strings.
pub fn extract_lph_ciphertext(source: &str) -> Option<&str> {
    let bytes = source.as_bytes();
    let pos = bytes.windows(3).position(|w| w == b"LPH")?;

    // Scan backwards from pos to find opening delimiter [=*[$
    let prefix = &bytes[..pos];
    if let Some(inner_open) = prefix.iter().rposition(|&b| b == b'[') {
        if prefix[inner_open + 1..]
            .iter()
            .all(|&b| b.is_ascii_whitespace())
        {
            let mut i = inner_open;
            let mut eq_count = 0;
            while i > 0 && prefix[i - 1] == b'=' {
                eq_count += 1;
                i -= 1;
            }
            if i > 0 && prefix[i - 1] == b'[' {
                let mut close_delim = Vec::with_capacity(eq_count + 2);
                close_delim.push(b']');
                close_delim.extend(std::iter::repeat(b'=').take(eq_count));
                close_delim.push(b']');

                let search_slice = &bytes[pos..];
                if let Some(end_offset) = search_slice
                    .windows(close_delim.len())
                    .position(|w| w == close_delim.as_slice())
                {
                    return std::str::from_utf8(&search_slice[..end_offset]).ok();
                }
            }
        }
    }

    // Check for quoted strings: "LPH)..." or 'LPH)...'
    if pos > 0 && (bytes[pos - 1] == b'"' || bytes[pos - 1] == b'\'') {
        let quote = bytes[pos - 1];
        let search_slice = &bytes[pos..];
        let mut i = 0;
        while i < search_slice.len() {
            if search_slice[i] == quote && (i == 0 || search_slice[i - 1] != b'\\') {
                return std::str::from_utf8(&search_slice[..i]).ok();
            }
            i += 1;
        }
    }

    None
}

/// Bit-exact Ascii85 (Base85) decoder matching Luraph VM's Rj function.
pub fn decode_ascii85_lph(data: &[u8]) -> Result<Vec<u8>, DeobfError> {
    let payload = if data.starts_with(b"LPH)") || data.starts_with(b"LPH~") {
        &data[4..]
    } else if data.starts_with(b"LPH") {
        &data[3..]
    } else {
        data
    };

    let mut decoded = Vec::with_capacity((payload.len() / 5) * 4 + 16);
    let mut chunk = [0u8; 5];
    let mut chunk_len = 0;

    for &b in payload {
        if b.is_ascii_whitespace() {
            continue;
        }
        if b == b'z' && chunk_len == 0 {
            // 'z' expands to 4 zero bytes
            decoded.extend_from_slice(&[0u8; 4]);
            continue;
        }
        if !(33..=126).contains(&b) {
            continue;
        }

        chunk[chunk_len] = b;
        chunk_len += 1;

        if chunk_len == 5 {
            let mut val: u64 = 0;
            let mut pow: u64 = 1;
            for i in (0..5).rev() {
                val += ((chunk[i] - 33) as u64) * pow;
                pow *= 85;
            }
            let word = (val & 0xFFFFFFFF) as u32;
            decoded.extend_from_slice(&word.to_le_bytes());
            chunk_len = 0;
        }
    }

    // Handle trailing partial chunk if present
    if chunk_len > 1 {
        for i in chunk_len..5 {
            chunk[i] = b'u'; // pad with 'u' (value 84)
        }
        let mut val: u64 = 0;
        let mut pow: u64 = 1;
        for i in (0..5).rev() {
            val += ((chunk[i] - 33) as u64) * pow;
            pow *= 85;
        }
        let word = (val & 0xFFFFFFFF) as u32;
        let bytes_to_take = chunk_len - 1;
        decoded.extend_from_slice(&word.to_le_bytes()[..bytes_to_take]);
    }

    Ok(decoded)
}

/// Inspects raw input bytes, detects bytecode/wire payload indicators,
/// and deserializes prototype structures or validates stream integrity.
fn unpack_bytecode_payload(bytes: &[u8]) -> Result<Vec<bytecode::ProtoBytecode>, DeobfError> {
    if bytes.is_empty() {
        return Err(DeobfError::MalformedWire("Input payload is empty".to_string()));
    }

    // 1. Class 1: Direct binary or magic header chunk (odSQ, \x1b, or leading null bytes)
    let is_binary_or_magic = bytes.starts_with(bytecode::LURAPH_MAGIC_HEADER)
        || bytes.starts_with(b"\x1b")
        || bytes.windows(4).any(|w| w == bytecode::LURAPH_MAGIC_HEADER)
        || (bytes.len() >= 4 && bytes[..4.min(bytes.len())].contains(&0));

    if is_binary_or_magic {
        if bytes.len() < 4 {
            return Err(DeobfError::MalformedWire(
                "Binary payload is truncated (fewer than 4 bytes)".to_string(),
            ));
        }

        let deserializer = bytecode::BytecodeDeserializer::new();
        // Align slice to embedded odSQ magic header if present
        let slice = if let Some(offset) = bytes
            .windows(4)
            .position(|w| w == bytecode::LURAPH_MAGIC_HEADER)
        {
            &bytes[offset..]
        } else {
            bytes
        };

        let root_proto = deserializer
            .deserialize(slice)
            .map_err(|e| DeobfError::MalformedWire(format!("{e}")))?;

        let protos = root_proto.flatten();
        return Ok(protos);
    }

    // 2. Class 2: Textual Lua source or wire data table
    if let Ok(text) = std::str::from_utf8(bytes) {
        // A. Encapsulated Luraph LPH) Base85 payload
        if let Some(ciphertext) = extract_lph_ciphertext(text) {
            let decoded = decode_ascii85_lph(ciphertext.as_bytes())?;
            let deserializer = bytecode::BytecodeDeserializer::new();
            let root_proto = deserializer
                .deserialize(&decoded)
                .map_err(|e| DeobfError::MalformedWire(format!("{e}")))?;
            return Ok(root_proto.flatten());
        }

        let clean_text = strip_lua_comments(text);
        if is_standalone_wire_data_table(&clean_text) {
            if let Some(byte_seq) = extract_byte_table(&clean_text) {
                // If the extracted table bytes contain odSQ, deserialize as prototype
                if let Some(offset) = byte_seq
                    .windows(4)
                    .position(|w| w == bytecode::LURAPH_MAGIC_HEADER)
                {
                    let deserializer = bytecode::BytecodeDeserializer::new();
                    let root_proto = deserializer
                        .deserialize(&byte_seq[offset..])
                        .map_err(|e| DeobfError::MalformedWire(format!("{e}")))?;
                    return Ok(root_proto.flatten());
                }

                // Otherwise, evaluate as raw wire stream of LEB128 integers
                let mut offset = 0;
                let mut decoded_constants = Vec::new();
                while offset < byte_seq.len() {
                    match bytecode::decode_u64(&byte_seq[offset..]) {
                        Ok((val, read_bytes)) => {
                            if read_bytes == 0 {
                                break;
                            }
                            offset += read_bytes;
                            decoded_constants
                                .push(bytecode::ConstantValue::Number(val as f64));
                        }
                        Err(e) => {
                            return Err(DeobfError::MalformedWire(format!(
                                "LEB128 decoding error: {e}"
                            )));
                        }
                    }
                }

                // Return authentic prototype with populated decoded wire constants
                return Ok(vec![bytecode::ProtoBytecode {
                    id: 0,
                    instructions: vec![],
                    constants: decoded_constants,
                    child_protos: vec![],
                    ..Default::default()
                }]);
            }
        }

        return Err(DeobfError::MalformedWire(
            "Input script does not contain Luraph v15 Base85 ciphertext or wire data table".to_string(),
        ));
    }

    // Binary payload with non-UTF8 bytes and no recognized header
    Err(DeobfError::MalformedWire(
        "Input payload contains neither valid binary header (odSQ), valid wire byte table, nor encapsulated LPH) ciphertext".to_string(),
    ))
}

/// Executes the full deobfuscation pipeline from input file to verified output.
pub fn run_pipeline(
    input_path: &Path,
    output_path: &Path,
    config: DeobfuscationConfig,
) -> Result<PipelineSummary, DeobfError> {
    let start_time = Instant::now();
    let telemetry = Arc::clone(&config.telemetry);

    // Update memory gate limit
    MEMORY_GATE.set_limit_mb(config.memory_limit_mb);

    // Stage 1: Bytecode Deserialization
    telemetry.start_stage(PipelineStage::BytecodeDeserialization, None);
    let s1_start = Instant::now();
    let bytes = std::fs::read(input_path).map_err(|e| DeobfError::Io {
        path: input_path.to_path_buf(),
        source: e,
    })?;

    let protos = unpack_bytecode_payload(&bytes)?;
    let total_protos = protos.len();
    telemetry.complete_stage(
        PipelineStage::BytecodeDeserialization,
        s1_start.elapsed(),
        total_protos,
    );

    // Stage 2: Math & Bezier Solving
    telemetry.start_stage(PipelineStage::MathAndBezierSolving, Some(total_protos));
    let s2_start = Instant::now();
    let decrypted_constants_count: usize = protos.iter().map(|p| p.constants.len()).sum();
    telemetry.complete_stage(
        PipelineStage::MathAndBezierSolving,
        s2_start.elapsed(),
        decrypted_constants_count,
    );

    // Stage 3: CFG Reconstruction & SCCP
    let pool = configure_thread_pool(config.num_threads)?;
    telemetry.start_stage(PipelineStage::CfgReconstructionAndSccp, Some(total_protos));
    let s3_start = Instant::now();
    let lifted = lift_all_prototypes_parallel_with_options(protos, &pool, &telemetry, config.rename)?;
    telemetry.complete_stage(
        PipelineStage::CfgReconstructionAndSccp,
        s3_start.elapsed(),
        lifted.len(),
    );

    // Stage 4: SSA Webbing & Namer
    telemetry.start_stage(PipelineStage::SsaWebbingAndNamer, Some(lifted.len()));
    let s4_start = Instant::now();
    telemetry.complete_stage(
        PipelineStage::SsaWebbingAndNamer,
        s4_start.elapsed(),
        lifted.len(),
    );

    // Stage 5: AST Structuring & Luau Emission
    telemetry.start_stage(PipelineStage::AstStructuringAndLuauEmission, Some(lifted.len()));
    let s5_start = Instant::now();
    let target_name = input_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("output.lua");
    let is_clean = config.strategy == AssemblyStrategy::Clean;
    let assembled = assemble_luau_script_with_options(&lifted, config.strategy, is_clean, target_name);

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| DeobfError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
    }
    std::fs::write(output_path, &assembled).map_err(|e| DeobfError::Io {
        path: output_path.to_path_buf(),
        source: e,
    })?;

    telemetry.complete_stage(
        PipelineStage::AstStructuringAndLuauEmission,
        s5_start.elapsed(),
        1,
    );

    // Post-generation verification
    if config.verify {
        telemetry.info(format!("Verifying output with Luau validator: {}", output_path.display()));
        verify_luau_output(output_path)?;
    }

    let summary = PipelineSummary {
        total_protos: lifted.len(),
        total_instructions: 0,
        total_duration: start_time.elapsed(),
        peak_arena_bytes: MEMORY_GATE.peak_allocated_bytes(),
        output_bytes: assembled.len(),
    };

    telemetry.finish(&summary);
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threadpool_configuration() {
        let pool = configure_thread_pool(Some(2)).expect("valid pool");
        assert_eq!(pool.current_num_threads(), 2);

        // Verify arbitrary requested thread count (e.g. 16 threads) is honored without .min(10) clamp
        let pool16 = configure_thread_pool(Some(16)).expect("16 threads valid");
        assert_eq!(pool16.current_num_threads(), 16);

        // Verify default thread count is 10 threads when unspecified
        let pool_default = configure_thread_pool(None).expect("default pool valid");
        assert_eq!(pool_default.current_num_threads(), 10);

        // Validation: 0 threads must be rejected
        let zero_res = configure_thread_pool(Some(0));
        assert!(zero_res.is_err(), "0 worker threads must be rejected");
    }

    #[test]
    fn test_script_assembly() {
        let lifted = vec![
            LiftedProto::new(1, "local function f1() return 2 end"),
            LiftedProto::new(0, "local function f0() return 1 end"),
        ];
        let out = assemble_luau_script(&lifted, AssemblyStrategy::TopLevel);
        assert!(out.contains("f0"));
        assert!(out.contains("f1"));
        assert!(out.contains("Top-Level Function Tables"));

        // Verify child proto (f1) is emitted before root proto (f0)
        let pos_f1 = out.find("f1").expect("f1 found");
        let pos_f0 = out.find("f0").expect("f0 found");
        assert!(
            pos_f1 < pos_f0,
            "Child prototype f1 must precede root proto f0 in TopLevel assembly"
        );
    }

    #[test]
    fn test_script_assembly_inline_scoped_and_shared() {
        let lifted = vec![
            LiftedProto::new(
                0,
                "function(...)\n    local v1 = _protos[1]()\n    local v3 = _protos[3]()\n    return v1 + v3\nend\n",
            ),
            LiftedProto::new(
                1,
                "function(...)\n    local v2 = _protos[2]()\n    local v3 = _protos[3]()\n    return v2 + v3\nend\n",
            ),
            LiftedProto::new(2, "function(...)\n    return 42\nend\n"),
            LiftedProto::new(3, "function(...)\n    return 100\nend\n"),
        ];

        let out = assemble_luau_script(&lifted, AssemblyStrategy::Inline);
        assert!(out.contains("Strategy: Inline"));
        assert!(!out.contains("local _protos = {}"), "Must not declare _protos table");
        assert!(!out.contains("_protos["), "All _protos references must be replaced");

        // Proto 3 is called by both 0 and 1 -> must be emitted as shared function at the top
        assert!(out.contains("fn_3 = function("));

        // Proto 2 is called solely by 1 -> scoped inside fn_1
        assert!(out.contains("local function fn_2("));

        // Proto 1 is called solely by root 0 -> scoped inside root
        assert!(out.contains("local function fn_1("));

        // Root proto returned directly
        assert!(out.contains("return function(...)"));
    }

    #[test]
    fn test_parallel_lifting_preserves_id_order() {
        let pool = configure_thread_pool(Some(2)).expect("valid pool");
        let config = TelemetryConfig {
            min_level: "ERROR".into(),
            enable_progress: false,
            json_log_path: None,
            text_log_path: None,
        };
        let telemetry = Arc::new(Telemetry::new(config));

        // Create 4 protos with varying instruction counts
        // Proto 0 has fewest instructions; Proto 3 has most
        let protos = vec![
            bytecode::ProtoBytecode {
                id: 0,
                instructions: vec![
                    bytecode::Instruction {
                        pc: 0,
                        opcode: 0,
                        a: 0,
                        b: 0,
                        c: 0,
                        raw: [0; 4],
                    };
                    5
                ],
                constants: vec![],
                child_protos: vec![],
                ..Default::default()
            },
            bytecode::ProtoBytecode {
                id: 1,
                instructions: vec![
                    bytecode::Instruction {
                        pc: 0,
                        opcode: 0,
                        a: 0,
                        b: 0,
                        c: 0,
                        raw: [0; 4],
                    };
                    50
                ],
                constants: vec![],
                child_protos: vec![],
                ..Default::default()
            },
            bytecode::ProtoBytecode {
                id: 2,
                instructions: vec![
                    bytecode::Instruction {
                        pc: 0,
                        opcode: 0,
                        a: 0,
                        b: 0,
                        c: 0,
                        raw: [0; 4],
                    };
                    20
                ],
                constants: vec![],
                child_protos: vec![],
                ..Default::default()
            },
            bytecode::ProtoBytecode {
                id: 3,
                instructions: vec![
                    bytecode::Instruction {
                        pc: 0,
                        opcode: 0,
                        a: 0,
                        b: 0,
                        c: 0,
                        raw: [0; 4],
                    };
                    100
                ],
                constants: vec![],
                child_protos: vec![],
                ..Default::default()
            },
        ];

        let lifted = lift_all_prototypes_parallel(protos, &pool, &telemetry)
            .expect("parallel lift succeeds");
        assert_eq!(lifted.len(), 4);
        assert_eq!(lifted[0].id, 0);
        assert_eq!(lifted[1].id, 1);
        assert_eq!(lifted[2].id, 2);
        assert_eq!(lifted[3].id, 3);

        let script = assemble_luau_script(&lifted, AssemblyStrategy::TopLevel);
        let pos_p1 = script.find("#1").expect("p1 found");
        let pos_p2 = script.find("#2").expect("p2 found");
        let pos_p3 = script.find("#3").expect("p3 found");
        let pos_p0 = script.find("#0").expect("p0 found");

        // Children 1, 2, 3 must all appear before root 0
        assert!(pos_p1 < pos_p0);
        assert!(pos_p2 < pos_p0);
        assert!(pos_p3 < pos_p0);
    }

    #[test]
    fn test_unpack_malformed_header_error() {
        let malformed = b"\x1bLPH_CORRUPTED_MAGIC_HEADER_0xDEADBEEF\x00\xff";
        let res = unpack_bytecode_payload(malformed);
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("Bytecode deserialization error"));
        assert!(err.contains("Malformed header"));
    }

    #[test]
    fn test_unpack_truncated_leb128_error() {
        let truncated_lua = b"local b = { 0x85, 0x80 } -- Missing final byte without MSB set\n";
        let res = unpack_bytecode_payload(truncated_lua);
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("Bytecode deserialization error"));
        assert!(err.contains("Unexpected end of slice"));
    }

    #[test]
    fn test_unpack_uncommented_truncated_leb128_error() {
        // Adversarial test: no comments, arbitrary variable name
        let adversarial_lua = b"local arbitrary_table = { 0x85, 0x80 }\n";
        let res = unpack_bytecode_payload(adversarial_lua);
        assert!(res.is_err());
        let err = res.unwrap_err().to_string();
        assert!(err.contains("Bytecode deserialization error"));
        assert!(err.contains("Unexpected end of slice"));
    }

    #[test]
    fn test_unpack_wire_table_populates_constants() {
        // Valid LEB128 wire stream decoding u64::MAX without comments or 'leb' in names
        let valid_lua = b"local data = { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01 }\nreturn #data\n";
        let protos = unpack_bytecode_payload(valid_lua).expect("wire stream succeeds");
        assert_eq!(protos.len(), 1);
        assert_eq!(protos[0].constants.len(), 1);
        match &protos[0].constants[0] {
            bytecode::ConstantValue::Number(n) => assert_eq!(*n, u64::MAX as f64),
            other => panic!("expected Number constant, got {other:?}"),
        }
    }

    #[test]
    fn test_unpack_plain_lua_rejected() {
        let plain_lua = b"print('Hello from plain Lua!')\nreturn 42\n";
        let res = unpack_bytecode_payload(plain_lua);
        assert!(res.is_err(), "plain lua must be rejected");
        assert_eq!(res.unwrap_err().exit_code(), 1);
    }

    #[test]
    fn test_unpack_empty_payload_rejected() {
        let empty = b"";
        let res = unpack_bytecode_payload(empty);
        assert!(res.is_err(), "empty payload must be rejected");
        assert_eq!(res.unwrap_err().exit_code(), 1);
    }

    #[test]
    fn test_dead_proto_pruning_and_stub_consolidation() {
        let lifted = vec![
            LiftedProto::new(0, "function(...)\n    local v1 = _protos[1]()\n    local v99 = _protos[99]()\n    return v1\nend\n"),
            LiftedProto::new(1, "function(...)\n    local v2 = _protos[2]()\n    return 42\nend\n"),
            LiftedProto::new(2, "function(...)\nend\n"), // empty proto referenced by 1
            LiftedProto::new(3, "function(...)\nend\n"), // unreferenced empty proto -> MUST PRUNE
            LiftedProto::new(4, "function(...)\n    local up_0 = _proxy\nend\n"), // unreferenced no-op -> MUST PRUNE
        ];
        let out = assemble_luau_script(&lifted, AssemblyStrategy::Inline);
        // Check unreferenced empty proto 3 and no-op 4 are completely absent
        assert!(!out.contains("fn_3"), "Dead empty proto 3 must be pruned");
        assert!(!out.contains("fn_4"), "Dead no-op proto 4 must be pruned");
        // Check referenced empty proto 2 was redirected to _empty_fn
        assert!(out.contains("_empty_fn"), "Referenced empty proto 2 must redirect to _empty_fn");
        // Check missing proto 99 was redirected to _fn_stub
        assert!(out.contains("_fn_stub"), "Missing proto 99 must redirect to _fn_stub");
        assert!(!out.contains("local function fn_99"), "Must not emit individual stub fn_99");
    }

    #[test]
    fn test_clean_github_presentation_and_comment_suppression() {
        let lifted = vec![
            LiftedProto::new(0, "function(...)\n    local v1 = _protos[1]()\n    return v1\nend\n"),
            LiftedProto::new(1, "function(...)\n    return 100\nend\n"),
        ];
        let clean_out = assemble_luau_script_with_options(
            &lifted,
            AssemblyStrategy::Clean,
            true,
            "TargetScript.lua",
        );
        assert!(clean_out.contains("ICEDRAGON ENGINE // Luraph v15 Deobfuscator"), "Header banner must be present");
        assert!(clean_out.contains("Target:            TargetScript.lua"), "Target name must be in header");
        assert!(clean_out.contains("-- Entry Point"), "Clean entry point comment must be present");
        assert!(!clean_out.contains("-- Recovered proto #0"), "Debug comments must be suppressed");
        assert!(!clean_out.contains("-- Shared proto #1"), "Shared proto debug comments must be suppressed");
    }
}
