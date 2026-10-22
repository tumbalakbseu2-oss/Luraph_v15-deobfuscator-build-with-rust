//! Comprehensive unit tests for Roblox semantic inference, linear scan interval
//! coloring, and Luau local budget limiter (<= 180 locals).

use std::collections::HashSet;
use bumpalo::Bump;
use luraph_deobf_rs::cfg::ir::IrExpr;
use luraph_deobf_rs::codegen::{LuauExpr, LuauStat};
use luraph_deobf_rs::namer::inference::{
    singularize, to_camel_case, HintPriority, RobloxSemanticInference,
};
use luraph_deobf_rs::namer::scope::{
    declared_here, limit_locals, split_block, unlocalize, LUAU_LOCAL_BUDGET,
};
use luraph_deobf_rs::namer::{IntervalNamer, NamerConfig};
use luraph_deobf_rs::ssa::{VariableWeb, WebId};

#[test]
fn test_datatype_semantic_inference() {
    let bump = Bump::new();

    // Vector3.new(...)
    let dt = bump.alloc(IrExpr::Global("Vector3"));
    let key = bump.alloc(IrExpr::String("new"));
    let callee = bump.alloc(IrExpr::TableGet { table: dt, key });
    let args: &[IrExpr] = bump.alloc_slice_clone(&[
        IrExpr::Number(1.0),
        IrExpr::Number(2.0),
        IrExpr::Number(3.0),
    ]);
    let call_expr = IrExpr::Call { callee, args };

    let hint = RobloxSemanticInference::infer_from_expr(&call_expr, 10, 0);
    assert!(hint.is_some(), "Vector3.new must yield semantic hint");
    let h = hint.unwrap();
    assert_eq!(h.name, "vector");
    assert_eq!(h.priority, HintPriority::Strong);

    // CFrame.Angles(...)
    let cf = bump.alloc(IrExpr::Global("CFrame"));
    let cf_key = bump.alloc(IrExpr::String("Angles"));
    let cf_callee = bump.alloc(IrExpr::TableGet { table: cf, key: cf_key });
    let cf_call = IrExpr::Call {
        callee: cf_callee,
        args: bump.alloc_slice_clone(&[IrExpr::Number(0.0)]),
    };
    let cf_hint = RobloxSemanticInference::infer_from_expr(&cf_call, 20, 0).unwrap();
    assert_eq!(cf_hint.name, "cframe");
    assert_eq!(cf_hint.priority, HintPriority::Strong);
}

#[test]
fn test_service_singleton_inference() {
    let bump = Bump::new();

    // game:GetService("Players")
    let game = bump.alloc(IrExpr::Global("game"));
    let get_svc = bump.alloc(IrExpr::String("GetService"));
    let callee = bump.alloc(IrExpr::TableGet { table: game, key: get_svc });
    let args: &[IrExpr] = bump.alloc_slice_clone(&[
        IrExpr::Global("game"),
        IrExpr::String("Players"),
    ]);
    let call = IrExpr::Call { callee, args };

    let hint = RobloxSemanticInference::infer_from_expr(&call, 5, 0).unwrap();
    assert_eq!(hint.name, "players");
    assert_eq!(hint.priority, HintPriority::Strong);

    // game:GetService("TweenService") -> tweenService
    let args2: &[IrExpr] = bump.alloc_slice_clone(&[
        IrExpr::Global("game"),
        IrExpr::String("TweenService"),
    ]);
    let call2 = IrExpr::Call { callee, args: args2 };
    let hint2 = RobloxSemanticInference::infer_from_expr(&call2, 6, 0).unwrap();
    assert_eq!(hint2.name, "tweenService");
    assert_eq!(hint2.priority, HintPriority::Strong);
}

#[test]
fn test_instance_new_inference() {
    let bump = Bump::new();

    // Instance.new("Part")
    let inst = bump.alloc(IrExpr::Global("Instance"));
    let new_k = bump.alloc(IrExpr::String("new"));
    let callee = bump.alloc(IrExpr::TableGet { table: inst, key: new_k });
    let args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::String("Part")]);
    let call = IrExpr::Call { callee, args };

    let hint = RobloxSemanticInference::infer_from_expr(&call, 1, 0).unwrap();
    assert_eq!(hint.name, "part");
    assert_eq!(hint.priority, HintPriority::Strong);

    // Instance.new("RemoteEvent")
    let args_re: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::String("RemoteEvent")]);
    let call_re = IrExpr::Call { callee, args: args_re };
    let hint_re = RobloxSemanticInference::infer_from_expr(&call_re, 2, 0).unwrap();
    assert_eq!(hint_re.name, "remoteEvent");
    assert_eq!(hint_re.priority, HintPriority::Strong);
}

#[test]
fn test_child_navigation_inference() {
    let bump = Bump::new();

    // inst:WaitForChild("HumanoidRootPart")
    let obj = bump.alloc(IrExpr::Register(1));
    let wait_k = bump.alloc(IrExpr::String("WaitForChild"));
    let callee = bump.alloc(IrExpr::TableGet { table: obj, key: wait_k });
    let args: &[IrExpr] = bump.alloc_slice_clone(&[
        IrExpr::Register(1),
        IrExpr::String("HumanoidRootPart"),
    ]);
    let call = IrExpr::Call { callee, args };

    let hint = RobloxSemanticInference::infer_from_expr(&call, 15, 0).unwrap();
    assert_eq!(hint.name, "humanoidRootPart");
    assert_eq!(hint.priority, HintPriority::Strong);
}

#[test]
fn test_standard_method_inference() {
    let bump = Bump::new();

    // inst:GetChildren()
    let obj = bump.alloc(IrExpr::Register(0));
    let m = bump.alloc(IrExpr::String("GetChildren"));
    let callee = bump.alloc(IrExpr::TableGet { table: obj, key: m });
    let args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
    let call = IrExpr::Call { callee, args };

    let hint = RobloxSemanticInference::infer_from_expr(&call, 1, 0).unwrap();
    assert_eq!(hint.name, "children");

    // signal:Connect(...)
    let sig_m = bump.alloc(IrExpr::String("Connect"));
    let sig_callee = bump.alloc(IrExpr::TableGet { table: obj, key: sig_m });
    let sig_call = IrExpr::Call { callee: sig_callee, args };
    let sig_hint = RobloxSemanticInference::infer_from_expr(&sig_call, 2, 0).unwrap();
    assert_eq!(sig_hint.name, "connection");
}

#[test]
fn test_multi_return_pcall() {
    let bump = Bump::new();

    let pcall_glob = bump.alloc(IrExpr::Global("pcall"));
    let args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(2)]);
    let call = IrExpr::Call { callee: pcall_glob, args };

    // Return index 0: ok
    let hint0 = RobloxSemanticInference::infer_from_expr(&call, 5, 0).unwrap();
    assert_eq!(hint0.name, "ok");
    assert_eq!(hint0.priority, HintPriority::Strong);

    // Return index 1: result
    let hint1 = RobloxSemanticInference::infer_from_expr(&call, 5, 1).unwrap();
    assert_eq!(hint1.name, "result");
    assert_eq!(hint1.priority, HintPriority::Strong);
}

#[test]
fn test_camel_case_sanitization_and_keywords() {
    assert_eq!(to_camel_case("UIFrame"), Some("uiFrame".to_string()));
    assert_eq!(to_camel_case("Part"), Some("part".to_string()));
    assert_eq!(to_camel_case("UICorner"), Some("uiCorner".to_string()));
    assert_eq!(to_camel_case("HumanoidRootPart"), Some("humanoidRootPart".to_string()));

    // Keyword collision escaping
    assert_eq!(to_camel_case("local"), Some("local_".to_string()));
    assert_eq!(to_camel_case("function"), Some("function_".to_string()));
    assert_eq!(to_camel_case("end"), Some("end_".to_string()));

    // Leading digits prefixing
    assert_eq!(to_camel_case("123item"), Some("n123item".to_string()));
}

#[test]
fn test_singularize() {
    assert_eq!(singularize("children"), Some("child".to_string()));
    assert_eq!(singularize("players"), Some("player".to_string()));
    assert_eq!(singularize("parts"), Some("part".to_string()));
    assert_eq!(singularize("entities"), Some("entity".to_string()));
}

#[test]
fn test_interval_namer_non_overlapping_reuse() {
    let namer = IntervalNamer::new(NamerConfig::default());

    let mut w1 = VariableWeb::new(WebId::new(0), 0);
    w1.live_interval = (0, 5);

    let mut w2 = VariableWeb::new(WebId::new(1), 1);
    w2.live_interval = (7, 12);

    let mut webs = vec![w1, w2];
    namer.assign_names(&mut webs);

    assert_eq!(webs[0].assigned_name, "v0");
    assert_eq!(
        webs[1].assigned_name, "v0",
        "Non-overlapping live intervals must reuse compact color v0"
    );
}

#[test]
fn test_interval_namer_overlapping_conflict_avoidance() {
    let namer = IntervalNamer::new(NamerConfig::default());

    let mut w1 = VariableWeb::new(WebId::new(0), 0);
    w1.live_interval = (0, 10);

    let mut w2 = VariableWeb::new(WebId::new(1), 1);
    w2.live_interval = (5, 15); // Overlaps with w1 in [5, 10]

    let mut webs = vec![w1, w2];
    namer.assign_names(&mut webs);

    assert_ne!(
        webs[0].assigned_name, webs[1].assigned_name,
        "Overlapping live intervals must NEVER receive the same variable name"
    );
    assert_eq!(webs[0].assigned_name, "v0");
    assert_eq!(webs[1].assigned_name, "v1");
}

#[test]
fn test_interval_namer_large_scale_scalability() {
    let namer = IntervalNamer::new(NamerConfig::default());

    let mut webs = Vec::new();
    // 500 webs spanning across 5,000 instructions
    for i in 0..500 {
        let mut w = VariableWeb::new(WebId::new(i as u32), (i % 256) as u16);
        let start = (i * 7) % 4500;
        let end = start + 30;
        w.live_interval = (start, end);
        webs.push(w);
    }

    let start_t = std::time::Instant::now();
    namer.assign_names(&mut webs);
    let elapsed = start_t.elapsed();

    assert!(
        elapsed.as_millis() < 50,
        "500-web linear scan must execute in < 50ms, took {:?}",
        elapsed
    );

    // Validate invariant: no two overlapping webs share a name
    for i in 0..webs.len() {
        for j in (i + 1)..webs.len() {
            if webs[i].live_interval.0 <= webs[j].live_interval.1
                && webs[j].live_interval.0 <= webs[i].live_interval.1
            {
                assert_ne!(
                    webs[i].assigned_name, webs[j].assigned_name,
                    "Collision detected between web {} and {} (both named {})",
                    i, j, webs[i].assigned_name
                );
            }
        }
    }
}

#[test]
fn test_difference_array_split_cuts_at_optimal_point() {
    let mut stmts = Vec::new();
    for i in 0..200 {
        let name = Box::leak(format!("v_{i}").into_boxed_str());
        stmts.push(LuauStat::LocalAssign {
            names: vec![name],
            values: vec![LuauExpr::Number(i as f64)],
        });
    }

    let split_res = split_block(stmts, 100);
    assert!(split_res.is_some(), "split_block must find a valid cut");

    let out = split_res.unwrap();
    let has_doblock = out.iter().any(|s| matches!(s, LuauStat::DoBlock(_)));
    assert!(has_doblock, "Result must contain at least one DoBlock");
}

#[test]
fn test_hoisted_variables_unlocalized() {
    let stmts = vec![
        LuauStat::LocalAssign {
            names: vec!["a"],
            values: vec![LuauExpr::Number(1.0)],
        },
        LuauStat::LocalAssign {
            names: vec!["b"],
            values: vec![LuauExpr::Number(2.0)],
        },
    ];

    let mut hoisted = HashSet::new();
    hoisted.insert("a");

    let unlocalized = unlocalize(stmts, &hoisted);
    assert_eq!(unlocalized.len(), 2);

    match &unlocalized[0] {
        LuauStat::Assign { targets, .. } => {
            assert_eq!(targets, &vec![LuauExpr::Identifier("a")]);
        }
        other => panic!("expected plain Assign, got {other:?}"),
    }

    match &unlocalized[1] {
        LuauStat::LocalAssign { names, .. } => {
            assert_eq!(names, &vec!["b"]);
        }
        other => panic!("expected LocalAssign, got {other:?}"),
    }
}

#[test]
fn test_limit_locals_respects_180_budget() {
    let mut stmts = Vec::new();
    for i in 0..250 {
        let name = Box::leak(format!("x_{i}").into_boxed_str());
        stmts.push(LuauStat::LocalAssign {
            names: vec![name],
            values: vec![LuauExpr::Number(i as f64)],
        });
    }

    let fitted = limit_locals(stmts, 0, LUAU_LOCAL_BUDGET);
    let direct_decls: usize = fitted.iter().map(|s| declared_here(s).len()).sum();
    assert!(direct_decls <= LUAU_LOCAL_BUDGET, "Direct declarations must not exceed 180");
}

#[test]
fn test_limit_locals_nested_scopes() {
    let mut inner = Vec::new();
    for i in 0..200 {
        let name = Box::leak(format!("inner_{i}").into_boxed_str());
        inner.push(LuauStat::LocalAssign {
            names: vec![name],
            values: vec![LuauExpr::Number(i as f64)],
        });
    }

    let if_stat = LuauStat::If {
        cond: LuauExpr::Boolean(true),
        then_branch: inner,
        elseif_branches: Vec::new(),
        else_branch: None,
    };

    let fitted = limit_locals(vec![if_stat], 0, LUAU_LOCAL_BUDGET);
    assert_eq!(fitted.len(), 1);

    if let LuauStat::If { then_branch, .. } = &fitted[0] {
        let direct_inner_decls: usize = then_branch.iter().map(|s| declared_here(s).len()).sum();
        assert!(direct_inner_decls <= LUAU_LOCAL_BUDGET, "Inner scope declarations must not exceed 180");
    } else {
        panic!("expected If statement");
    }
}

#[test]
fn test_adversarial_dense_crossing_locals_guarantee_180_budget() {
    let mut stmts = Vec::new();
    // 300 variables in a single scope block
    for i in 0..300 {
        let name = Box::leak(format!("dense_{i}").into_boxed_str());
        stmts.push(LuauStat::LocalAssign {
            names: vec![name],
            values: vec![LuauExpr::Number(i as f64)],
        });
    }

    let fitted = limit_locals(stmts, 0, LUAU_LOCAL_BUDGET);
    let direct_decls: usize = fitted.iter().map(|s| declared_here(s).len()).sum();
    assert!(
        direct_decls <= LUAU_LOCAL_BUDGET,
        "Direct declarations must strictly not exceed 180 (got {})",
        direct_decls
    );
}

