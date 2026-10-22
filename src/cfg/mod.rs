//! Control Flow Graph construction, SCCP symbolic evaluation, and structurer.
//!
//! Provides the core Milestone 3 pipeline for basic block graph lifting,
//! sparse conditional constant propagation, control-flow unflattening,
//! dominator analysis, and high-level region structuring.

pub mod builder;
pub mod dominator;
pub mod ir;
pub mod sccp;
pub mod structurer;
pub mod unflatten;

pub use builder::{build_cfg, merge_equivalent, prune_unreachable, thread_empty, LeaderAnalyzer};
pub use dominator::{DominatorTree, LoopExtractor, NaturalLoop, PostDominatorTree};
pub use ir::{
    BasicBlock, BinOp, BlockId, ControlFlowGraph, Edge, EdgeType, IrExpr, IrInst,
    Literal, LoopKind, StructuredRegion, Terminator, UnOp,
};
pub use sccp::{AbstractInterpreter, LatticeValue, SccpEngine, VmState};
pub use structurer::{
    emit_luau, emit_luau_structured, expr_to_string, lower_expr, lower_inst, lower_region,
    RegionStructurer,
};
pub use unflatten::{DispatcherInfo, Unflattener};

use bumpalo::Bump;

/// Lifts a bytecode prototype into structured Luau source code with default options.
pub fn lift_prototype<'arena>(
    proto: &crate::bytecode::ProtoBytecode,
    bump: &'arena Bump,
) -> String {
    lift_prototype_with_options(proto, bump, true)
}

/// Lifts a bytecode prototype into structured Luau source code with configurable renaming.
pub fn lift_prototype_with_options<'arena>(
    proto: &crate::bytecode::ProtoBytecode,
    bump: &'arena Bump,
    rename: bool,
) -> String {
    let mut output = String::from("function(...)\n");

    if proto.instructions.is_empty() {
        output.push_str("end\n");
        return output;
    }

    // Stage 1: Build CFG & Initial Normalization
    let mut cfg = build_cfg(proto, bump);
    prune_unreachable(&mut cfg);
    thread_empty(&mut cfg);

    // Filtered upvalues: only emit declarations for genuinely referenced upvalues!
    let referenced_upvalues = crate::ssa::dataflow::collect_referenced_upvalues(&cfg);

    if !referenced_upvalues.is_empty() {
        let upval_names: Vec<String> = referenced_upvalues
            .iter()
            .map(|u| format!("up_{u}"))
            .collect();
        for chunk in upval_names.chunks(10) {
            let inits = vec!["_proxy"; chunk.len()].join(", ");
            output.push_str(&format!("    local {} = {}\n", chunk.join(", "), inits));
        }
    }

    // Bind function parameters to registers
    let safe_num_params = (proto.num_params as usize).min(30);
    if safe_num_params > 0 {
        let param_names: Vec<String> = (0..safe_num_params).map(|p| format!("r{p}")).collect();
        for chunk in param_names.chunks(10) {
            output.push_str(&format!("    local {} = ...\n", chunk.join(", ")));
        }
        for chunk in param_names.chunks(10) {
            let guards: Vec<String> = chunk.iter().map(|p| format!("if {p} == nil then {p} = _proxy end")).collect();
            output.push_str(&format!("    {}\n", guards.join(" ")));
        }
    }

    // In Luraph v15, constants are evaluated and inlined directly into expressions by constant_to_ir_expr.
    // Register definitions are declared as local variables in-place by lower_inst.
    // Only parameters and captured upvalues are declared at the function prologue.
    let outer_locals = referenced_upvalues.len() + safe_num_params;

    // Stage 2: Sparse Conditional Constant Propagation
    let mut sccp = SccpEngine::new(&mut cfg);
    sccp.run();

    // Stage 3: Control-Flow Unflattening
    if let Some(disp) = Unflattener::detect_dispatcher(&cfg) {
        Unflattener::unflatten(&mut cfg, &disp);
    }

    // Stage 4: Secondary Normalization & Partition Refinement
    thread_empty(&mut cfg);
    merge_equivalent(&mut cfg);

    // Stage 4.5: SSA Variable Web Coalescence & Dead Store Elimination (Milestone 4)
    let mut ssa_ctx = crate::ssa::analyze_ssa(&cfg, proto.num_params);
    crate::ssa::eliminate_dead_stores(&mut cfg, &ssa_ctx);

    // Stage 4.6: Roblox Semantic Naming & Live-Range Interval Coloring (Milestone 4)
    if rename {
        crate::namer::run_namer(&mut ssa_ctx, &cfg);
    }

    // Stage 5: Region Structuring & Emission with Luau Local Budget Limiting (Milestone 4)
    let mut structurer = RegionStructurer::new(&cfg);
    let region = structurer.structure();
    let mut body = emit_luau_structured(&region, &cfg, &ssa_ctx, bump, outer_locals, 1);

    if rename {
        body = RobloxSemanticNamer::rename_source(&body);
    }

    output.push_str(&body);
    output.push_str("end\n");
    output
}

/// AST & source level Roblox semantic identifier inference pass.
///
/// Refines register and upvalue names into idiomatic Roblox identifiers:
/// - Service singletons: `Players`, `TweenService`, `RunService`, `UserInputService`,
///   `ReplicatedStorage`, `HttpService`, `Workspace`, `Lighting`, `CoreGui`, etc.
/// - Core player entities: `LocalPlayer`, `mouse`, `character`
/// - Common API signals and methods: `connection`, `callback`, `children`
pub struct RobloxSemanticNamer;

impl RobloxSemanticNamer {
    /// Applies semantic Roblox naming over generated Luau code.
    pub fn rename_source(source: &str) -> String {
        let mut replacements: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        for line in source.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("local ") {
                if let Some((var_part, expr_part)) = rest.split_once('=') {
                    let var = var_part.trim();
                    let expr = expr_part.trim();

                    if (var.starts_with('r') && var[1..].chars().all(|c| c.is_ascii_digit()))
                        || (var.starts_with("up_") && var[3..].chars().all(|c| c.is_ascii_digit()))
                    {
                        if let Some(semantic_name) = Self::infer_name_from_expr(expr, &replacements) {
                            replacements.insert(var.to_string(), semantic_name);
                        }
                    }
                }
            }
        }

        if replacements.is_empty() {
            return source.to_string();
        }

        Self::apply_replacements(source, &replacements)
    }

    fn infer_name_from_expr(expr: &str, known_vars: &std::collections::HashMap<String, String>) -> Option<String> {
        if let Some(pos) = expr.find("GetService(") {
            let args = &expr[pos + "GetService(".len()..];
            if let Some(end_paren) = args.find(')') {
                let arg = args[..end_paren].trim().trim_matches(|c| c == '"' || c == '\'');
                let service_name = match arg {
                    "Players" => "Players",
                    "TweenService" => "TweenService",
                    "RunService" => "RunService",
                    "UserInputService" => "UserInputService",
                    "ReplicatedStorage" => "ReplicatedStorage",
                    "HttpService" => "HttpService",
                    "Workspace" => "Workspace",
                    "Lighting" => "Lighting",
                    "CoreGui" => "CoreGui",
                    "SoundService" => "SoundService",
                    "StarterGui" => "StarterGui",
                    "TeleportService" => "TeleportService",
                    other if !other.is_empty() => other,
                    _ => "",
                };
                if !service_name.is_empty() {
                    return Some(service_name.to_string());
                }
            }
        }

        if expr.contains(".LocalPlayer") {
            return Some("LocalPlayer".to_string());
        }

        if expr.contains(":GetMouse()") {
            return Some("mouse".to_string());
        }

        if expr.contains(".Character") && !expr.contains(".CharacterAdded") {
            return Some("character".to_string());
        }

        if expr.contains("WaitForChild(\"HumanoidRootPart\")") || expr.contains("FindFirstChild(\"HumanoidRootPart\")") {
            return Some("humanoidRootPart".to_string());
        }
        if expr.contains("WaitForChild(\"Humanoid\")") || expr.contains("FindFirstChild(\"Humanoid\")") {
            return Some("humanoid".to_string());
        }

        for (var, _) in known_vars {
            if expr.starts_with(var) && expr[var.len()..].starts_with(".LocalPlayer") {
                return Some("LocalPlayer".to_string());
            }
        }

        None
    }

    fn apply_replacements(source: &str, replacements: &std::collections::HashMap<String, String>) -> String {
        let mut out = String::with_capacity(source.len());
        for line in source.lines() {
            let mut modified_line = line.to_string();
            for (from, to) in replacements {
                modified_line = replace_word(&modified_line, from, to);
            }
            out.push_str(&modified_line);
            out.push('\n');
        }
        out
    }
}

/// Helper function to replace exact whole words (identifiers) in a string.
fn replace_word(text: &str, target: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remainder = text;

    while let Some(pos) = remainder.find(target) {
        let before_ok = if pos == 0 {
            true
        } else {
            let prev_char = remainder[..pos].chars().last().unwrap();
            !prev_char.is_alphanumeric() && prev_char != '_'
        };

        let after_pos = pos + target.len();
        let after_ok = if after_pos >= remainder.len() {
            true
        } else {
            let next_char = remainder[after_pos..].chars().next().unwrap();
            !next_char.is_alphanumeric() && next_char != '_'
        };

        if before_ok && after_ok {
            result.push_str(&remainder[..pos]);
            result.push_str(replacement);
            remainder = &remainder[after_pos..];
        } else {
            result.push_str(&remainder[..pos + target.len()]);
            remainder = &remainder[pos + target.len()..];
        }
    }
    result.push_str(remainder);
    result
}

