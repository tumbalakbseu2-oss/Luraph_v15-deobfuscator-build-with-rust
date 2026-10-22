//! Milestone 4 Adversarial Test Suite
//!
//! Empirical stress tests for SSA Variable Web Analysis, Reaching Definitions,
//! Disjoint-Set Union-Find coalescence, Open Upvalue tracking,
//! Semantic Hint Priority Lattice resolution, and Linear Scan Interval Coloring.

use std::collections::{HashMap, HashSet};
use std::time::Instant;
use bumpalo::Bump;

use luraph_deobf_rs::cfg::ir::{
    BasicBlock, BlockId, ControlFlowGraph, IrExpr, IrInst, Terminator, BinOp,
};
use luraph_deobf_rs::codegen::{LuauExpr, LuauStat};
use luraph_deobf_rs::namer::inference::{
    to_camel_case, HintPriority, SemanticHint, LUAU_KEYWORDS,
};
use luraph_deobf_rs::namer::scope::{
    declared_here, limit_locals,
};
use luraph_deobf_rs::namer::{IntervalNamer, NamerConfig};
use luraph_deobf_rs::ssa::{
    analyze_ssa, eliminate_dead_stores, run_reaching_definitions,
    DefSite, ProgramPoint, UseSite, VariableWeb, WebId,
};

// ============================================================================
// 1. SSA HIGH VARIABLE DENSITY & COMPLEX LOOP BACKEDGES
// ============================================================================

#[test]
fn test_adv_ssa_high_density_loop_backedges_nested() {
    // Construct a deeply nested loop structure with 5 nested loop tiers
    // and 32 virtual registers modified conditionally in the innermost loop
    // and propagated backwards to outer loop headers.
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Entry block: initializes 32 registers
    let mut entry = BasicBlock::new(BlockId(0), (0, 31), Terminator::Goto(BlockId(1)));
    for r in 0..32u16 {
        entry.instructions.push(IrInst::Assign {
            dest: r,
            src: IrExpr::Number(r as f64),
        });
    }
    cfg.add_block(entry);

    // Create 5 nested loop headers: Block 1, 2, 3, 4, 5
    // Loop 5 is the inner body which modifies registers and branches back or exits
    for level in 1..=4u32 {
        let b = BasicBlock::new(
            BlockId(level),
            (level as usize * 100, level as usize * 100),
            Terminator::Branch {
                cond: IrExpr::Boolean(true),
                true_target: BlockId(level + 1),
                false_target: BlockId(10 + level), // exit path
            },
        );
        cfg.add_block(b);
    }

    // Inner loop body: Block 5
    // Conditionally updates registers 0..16, and loops back to Block 5 or Block 3
    let mut inner = BasicBlock::new(
        BlockId(5),
        (500, 532),
        Terminator::Branch {
            cond: IrExpr::Boolean(true),
            true_target: BlockId(5), // inner backedge
            false_target: BlockId(6), // exit inner loop
        },
    );
    for r in 0..16u16 {
        let prev_r = bump.alloc(IrExpr::Register(r));
        let one = bump.alloc(IrExpr::Number(1.0));
        inner.instructions.push(IrInst::Assign {
            dest: r,
            src: IrExpr::Binary {
                op: BinOp::Add,
                left: prev_r,
                right: one,
            },
        });
    }
    cfg.add_block(inner);

    // Block 6: outer backedge back to Block 2 or exit to Block 7
    let b6 = BasicBlock::new(
        BlockId(6),
        (600, 600),
        Terminator::Branch {
            cond: IrExpr::Boolean(false),
            true_target: BlockId(2), // outer backedge
            false_target: BlockId(7),
        },
    );
    cfg.add_block(b6);

    // Block 7: terminal merge block reading all 32 registers
    let mut uses = Vec::new();
    for r in 0..32u16 {
        uses.push(IrExpr::Register(r));
    }
    let ret_slice: &[IrExpr] = bump.alloc_slice_clone(&uses);
    let b7 = BasicBlock::new(BlockId(7), (700, 700), Terminator::Return(ret_slice));
    cfg.add_block(b7);

    // Also add the exit blocks 11..14 targeting Block 7
    for level in 1..=4u32 {
        let b = BasicBlock::new(BlockId(10 + level), (800 + level as usize, 800 + level as usize), Terminator::Goto(BlockId(7)));
        cfg.add_block(b);
    }

    cfg.recompute_successors();

    let start = Instant::now();
    let ssa = analyze_ssa(&cfg, 0);
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 200, "High density loop backedge analysis took {:?}", elapsed);
    assert!(!ssa.webs.is_empty(), "Webs must be generated");

    // Registers 0..16 had multiple definitions across loop backedges and must be coalesced into single webs
    for r in 0..16u16 {
        let entry_def = DefSite::inst(BlockId(0), r as usize, r);
        let inner_def = DefSite::inst(BlockId(5), r as usize, r);

        let w_entry = ssa.def_to_web.get(&entry_def).expect("entry def mapped");
        let w_inner = ssa.def_to_web.get(&inner_def).expect("inner def mapped");

        assert_eq!(
            w_entry, w_inner,
            "Register {} across loop backedges must coalesce into the same VariableWeb",
            r
        );

        // Verify the use at the return block points to this coalesced web
        let use_site = UseSite::term(BlockId(7), r);
        assert_eq!(
            ssa.use_to_web.get(&use_site),
            Some(&Some(*w_entry)),
            "Return block use of register {} must resolve to the coalesced web",
            r
        );
    }
}

#[test]
fn test_adv_ssa_500_reaching_definitions_across_irreducible_dispatcher() {
    // 50 blocks arranged as an obfuscated dispatcher with 500 total definitions
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let num_blocks = 50;
    // Dispatcher header: Block 0
    let head = BasicBlock::new(
        BlockId(0),
        (0, 0),
        Terminator::Branch {
            cond: IrExpr::Register(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );
    cfg.add_block(head);

    for i in 1..num_blocks {
        let bid = BlockId(i as u32);
        let next_bid = if i + 1 < num_blocks {
            BlockId((i + 1) as u32)
        } else {
            BlockId(0) // loop back to dispatcher
        };

        let mut b = BasicBlock::new(
            bid,
            (i * 10, i * 10 + 9),
            if i % 5 == 0 {
                Terminator::Goto(BlockId(0))
            } else {
                Terminator::Goto(next_bid)
            },
        );

        // 10 register definitions per block
        for k in 0..10u16 {
            let reg = (i as u16 * 10 + k) % 64;
            b.instructions.push(IrInst::Assign {
                dest: reg,
                src: IrExpr::Number((i * 100 + k as usize) as f64),
            });
        }
        cfg.add_block(b);
    }

    cfg.recompute_successors();

    let start = Instant::now();
    let dataflow = run_reaching_definitions(&cfg, 0);
    let ssa = analyze_ssa(&cfg, 0);
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 500, "Dispatcher reaching definitions took {:?}", elapsed);
    assert!(dataflow.in_states.contains_key(&BlockId(0)));
    assert!(!ssa.webs.is_empty());
}

#[test]
fn test_adv_ssa_disjoint_def_use_webs_strict_isolation() {
    // A single register R(10) is used in 4 completely separate, sequential phases:
    // Phase 1 (Block 0): Def R(10) -> Use R(10) -> dead
    // Phase 2 (Block 1): Def R(10) -> Use R(10) -> dead
    // Phase 3 (Block 2): Def R(10) -> Use R(10) -> dead
    // Phase 4 (Block 3): Def R(10) -> Use R(10) -> dead
    // Each phase strictly overwrites R(10) before any reading.
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    for i in 0..4u32 {
        let bid = BlockId(i);
        let next_term = if i < 3 {
            Terminator::Goto(BlockId(i + 1))
        } else {
            Terminator::Return(bump.alloc_slice_clone(&[]))
        };

        let mut b = BasicBlock::new(bid, (i as usize * 10, i as usize * 10 + 5), next_term);

        // Statement 0: Def R(10)
        b.instructions.push(IrInst::Assign {
            dest: 10,
            src: IrExpr::Number(i as f64 * 100.0),
        });

        // Statement 1: Use R(10) in another register
        b.instructions.push(IrInst::Assign {
            dest: 20 + i as u16,
            src: IrExpr::Register(10),
        });

        cfg.add_block(b);
    }

    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);

    let d0 = DefSite::inst(BlockId(0), 0, 10);
    let d1 = DefSite::inst(BlockId(1), 0, 10);
    let d2 = DefSite::inst(BlockId(2), 0, 10);
    let d3 = DefSite::inst(BlockId(3), 0, 10);

    let w0 = *ssa.def_to_web.get(&d0).expect("d0 mapped");
    let w1 = *ssa.def_to_web.get(&d1).expect("d1 mapped");
    let w2 = *ssa.def_to_web.get(&d2).expect("d2 mapped");
    let w3 = *ssa.def_to_web.get(&d3).expect("d3 mapped");

    // All four webs MUST be strictly disjoint!
    let mut unique_webs = HashSet::new();
    unique_webs.insert(w0);
    unique_webs.insert(w1);
    unique_webs.insert(w2);
    unique_webs.insert(w3);

    assert_eq!(
        unique_webs.len(),
        4,
        "Sequential non-overlapping lifecycles of register R(10) MUST produce 4 distinct WebIds, got: {:?}",
        unique_webs
    );

    // Each use must point to its respective web
    let u0 = UseSite::inst(BlockId(0), 1, 10);
    let u1 = UseSite::inst(BlockId(1), 1, 10);
    let u2 = UseSite::inst(BlockId(2), 1, 10);
    let u3 = UseSite::inst(BlockId(3), 1, 10);

    assert_eq!(ssa.use_to_web.get(&u0), Some(&Some(w0)));
    assert_eq!(ssa.use_to_web.get(&u1), Some(&Some(w1)));
    assert_eq!(ssa.use_to_web.get(&u2), Some(&Some(w2)));
    assert_eq!(ssa.use_to_web.get(&u3), Some(&Some(w3)));

    // Verify live intervals are strictly disjoint and monotonic
    let web0 = ssa.get_web(w0).unwrap();
    let web1 = ssa.get_web(w1).unwrap();
    let web2 = ssa.get_web(w2).unwrap();
    let web3 = ssa.get_web(w3).unwrap();

    assert!(web0.live_interval.1 < web1.live_interval.0, "Interval 0 must end before Interval 1 starts");
    assert!(web1.live_interval.1 < web2.live_interval.0, "Interval 1 must end before Interval 2 starts");
    assert!(web2.live_interval.1 < web3.live_interval.0, "Interval 2 must end before Interval 3 starts");
}

// ============================================================================
// 2. DEEP CLOSURE NESTING & OPEN UPVALUE CAPTURE CHAINS
// ============================================================================

#[test]
fn test_adv_ssa_deep_closure_nesting_and_upvalue_capture_chains() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // Stmt 0: R(5) = 1.0 (Def 1)
    // Stmt 1: R(10) = Closure(proto 1, upvals: [R(5)]) -> Captures R(5)
    // Stmt 2: R(5) = 2.0 (Def 2: mutated while upvalue is actively open!)
    // Stmt 3: R(11) = Closure(proto 2, upvals: [R(5)]) -> Nested capture
    // Stmt 4: R(5) = 3.0 (Def 3: mutated again)
    // Stmt 5: CloseUpvalue(5) -> Explicitly closes open upvalue!
    // Stmt 6: R(5) = 4.0 (Def 4: after close! New distinct local variable!)
    // Stmt 7: R(20) = R(5) (Use 4)
    // Terminator: Return
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (0, 7),
        Terminator::Return(bump.alloc_slice_clone(&[])),
    );

    // Stmt 0: Def 1
    b0.instructions.push(IrInst::Assign {
        dest: 5,
        src: IrExpr::Number(1.0),
    });

    // Stmt 1: Closure 1 captures R(5)
    let upval1: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(5)]);
    b0.instructions.push(IrInst::Assign {
        dest: 10,
        src: IrExpr::Closure {
            proto_id: 1,
            upvalues: upval1,
        },
    });

    // Stmt 2: Def 2
    b0.instructions.push(IrInst::Assign {
        dest: 5,
        src: IrExpr::Number(2.0),
    });

    // Stmt 3: Closure 2 captures R(5)
    let upval2: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(5)]);
    b0.instructions.push(IrInst::Assign {
        dest: 11,
        src: IrExpr::Closure {
            proto_id: 2,
            upvalues: upval2,
        },
    });

    // Stmt 4: Def 3
    b0.instructions.push(IrInst::Assign {
        dest: 5,
        src: IrExpr::Number(3.0),
    });

    // Stmt 5: CloseUpvalue
    b0.instructions.push(IrInst::CloseUpvalue(5));

    // Stmt 6: Def 4
    b0.instructions.push(IrInst::Assign {
        dest: 5,
        src: IrExpr::Number(4.0),
    });

    // Stmt 7: Use
    b0.instructions.push(IrInst::Assign {
        dest: 20,
        src: IrExpr::Register(5),
    });

    cfg.add_block(b0);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);

    let d0 = DefSite::inst(BlockId(0), 0, 5);
    let d2 = DefSite::inst(BlockId(0), 2, 5);
    let d4 = DefSite::inst(BlockId(0), 4, 5);
    let d6 = DefSite::inst(BlockId(0), 6, 5);

    let w0 = *ssa.def_to_web.get(&d0).expect("d0 mapped");
    let w2 = *ssa.def_to_web.get(&d2).expect("d2 mapped");
    let w4 = *ssa.def_to_web.get(&d4).expect("d4 mapped");
    let w6 = *ssa.def_to_web.get(&d6).expect("d6 mapped");

    // Def 1, Def 2, Def 3 occurred while the upvalue was open, so they MUST coalesce!
    assert_eq!(w0, w2, "Def 1 and Def 2 while upvalue is open must coalesce");
    assert_eq!(w2, w4, "Def 2 and Def 3 while upvalue is open must coalesce");

    let upval_web = ssa.get_web(w0).unwrap();
    assert!(upval_web.is_upvalue, "Coalesced open upvalue web must have is_upvalue == true");

    // Def 4 occurred AFTER CloseUpvalue(5), so it MUST NOT coalesce with the upvalue web!
    assert_ne!(
        w0, w6,
        "Definition after CloseUpvalue must be partitioned into a distinct WebId"
    );

    let u7 = UseSite::inst(BlockId(0), 7, 5);
    assert_eq!(
        ssa.use_to_web.get(&u7),
        Some(&Some(w6)),
        "Use after CloseUpvalue must resolve to the post-close WebId"
    );
}

#[test]
fn test_adv_ssa_synthetic_capture_uninitialized_upvalue() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // Stmt 0: Closure captures R(15) which has NEVER been defined in this prototype!
    // Stmt 1: R(20) = R(15) -> use of uninitialized/synthetic captured register
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (0, 1),
        Terminator::Return(bump.alloc_slice_clone(&[])),
    );

    let upval: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(15)]);
    b0.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Closure {
            proto_id: 10,
            upvalues: upval,
        },
    });

    b0.instructions.push(IrInst::Assign {
        dest: 20,
        src: IrExpr::Register(15),
    });

    cfg.add_block(b0);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);

    // Verify synthetic capture was registered
    let syn_site = DefSite::synthetic_capture(ProgramPoint::Stmt(BlockId(0), 0), 15);
    let syn_web_id = ssa.def_to_web.get(&syn_site).expect("Synthetic capture must be assigned a WebId");

    let syn_web = ssa.get_web(*syn_web_id).unwrap();
    assert!(syn_web.is_upvalue, "Synthetic capture web must have is_upvalue == true");

    // The subsequent use must resolve to this synthetic web
    let u1 = UseSite::inst(BlockId(0), 1, 15);
    assert_eq!(
        ssa.use_to_web.get(&u1),
        Some(&Some(*syn_web_id)),
        "Use of uninitialized captured register must map to the synthetic capture WebId"
    );
}

#[test]
fn test_adv_ssa_dead_store_elimination_preserves_captured_upvalues() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // Stmt 0: R(0) = 100.0 (Def pure, never used, NOT captured -> SHOULD BE ELIMINATED)
    // Stmt 1: R(1) = 200.0 (Def pure, never used locally, BUT captured by closure -> MUST BE PRESERVED)
    // Stmt 2: R(2) = Closure(proto 1, upvals: [R(1)])
    // Stmt 3: Call print() (Side effect -> MUST BE PRESERVED)
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (0, 3),
        Terminator::Return(bump.alloc_slice_clone(&[])),
    );

    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(100.0),
    });

    b0.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(200.0),
    });

    let upval: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(1)]);
    b0.instructions.push(IrInst::Assign {
        dest: 2,
        src: IrExpr::Closure {
            proto_id: 1,
            upvalues: upval,
        },
    });

    b0.instructions.push(IrInst::CallStmt {
        callee: IrExpr::Global("print"),
        args: bump.alloc_slice_clone(&[]),
        ret_dest: None,
        ret_count: 0,
    });

    cfg.add_block(b0);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);
    let eliminated = eliminate_dead_stores(&mut cfg, &ssa);

    assert_eq!(eliminated, 1, "Exactly one dead store (R(0)) should be eliminated");

    let b0_final = cfg.blocks.get(&BlockId(0)).unwrap();
    assert_eq!(b0_final.instructions[0], IrInst::Nop, "Stmt 0 must be replaced with Nop");
    assert_ne!(b0_final.instructions[1], IrInst::Nop, "Stmt 1 (captured upvalue) must NOT be eliminated");
    assert_ne!(b0_final.instructions[2], IrInst::Nop, "Stmt 2 (closure creation) must NOT be eliminated");
    assert_ne!(b0_final.instructions[3], IrInst::Nop, "Stmt 3 (side-effect call) must NOT be eliminated");
}

// ============================================================================
// 3. NAMER SEMANTIC HINT PRIORITY LATTICE RESOLUTION
// ============================================================================

#[test]
fn test_adv_namer_semantic_hint_lattice_priority_resolution() {
    let h_default = SemanticHint::new("v1", HintPriority::Default, 10);
    let h_weak = SemanticHint::new("str", HintPriority::Weak, 20);
    let h_label = SemanticHint::new("health", HintPriority::Label, 30);
    let h_strong_early = SemanticHint::new("players", HintPriority::Strong, 5);
    let h_strong_late = SemanticHint::new("character", HintPriority::Strong, 25);

    // 1. Weak beats Default
    assert_eq!(h_default.clone().merge(h_weak.clone()), h_weak);
    assert_eq!(h_weak.clone().merge(h_default.clone()), h_weak);

    // 2. Label beats Weak
    assert_eq!(h_weak.clone().merge(h_label.clone()), h_label);
    assert_eq!(h_label.clone().merge(h_weak.clone()), h_label);

    // 3. Strong beats Label
    assert_eq!(h_label.clone().merge(h_strong_early.clone()), h_strong_early);
    assert_eq!(h_strong_early.clone().merge(h_label.clone()), h_strong_early);

    // 4. Equal priority tie-break: earlier origin PC wins
    let tie_result_1 = h_strong_early.clone().merge(h_strong_late.clone());
    let tie_result_2 = h_strong_late.clone().merge(h_strong_early.clone());
    assert_eq!(tie_result_1.name, "players", "Earlier definition origin (pc 5) must win over pc 25");
    assert_eq!(tie_result_2.name, "players", "Earlier definition origin (pc 5) must win regardless of merge order");

    // 5. Idempotency: x.merge(x) == x
    assert_eq!(h_strong_early.clone().merge(h_strong_early.clone()), h_strong_early);
    assert_eq!(h_label.clone().merge(h_label.clone()), h_label);
}

#[test]
fn test_adv_namer_luau_keyword_evasion_and_sanitization() {
    // Test that all 25 Luau keywords are properly detected and escaped
    for &kw in LUAU_KEYWORDS {
        let escaped = to_camel_case(kw).expect("Escaped keyword");
        assert_ne!(escaped, kw, "Keyword '{}' must not remain identical", kw);
        assert!(escaped.ends_with('_'), "Keyword '{}' should be escaped with suffix '_', got '{}'", kw, escaped);
    }

    // Test interval namer disambiguation when semantic hints collide with keywords
    let mut webs = vec![
        VariableWeb {
            id: WebId(0),
            original_reg: 0,
            assigned_name: String::new(),
            live_interval: (0, 10),
            is_upvalue: false,
            is_loop_var: false,
            defs: vec![],
            uses: vec![],
            semantic_hint: Some(SemanticHint::new("local", HintPriority::Strong, 0)),
        },
        VariableWeb {
            id: WebId(1),
            original_reg: 1,
            assigned_name: String::new(),
            live_interval: (2, 8),
            is_upvalue: false,
            is_loop_var: false,
            defs: vec![],
            uses: vec![],
            semantic_hint: Some(SemanticHint::new("function", HintPriority::Strong, 2)),
        },
    ];

    let namer = IntervalNamer::new(NamerConfig::default());
    namer.assign_names(&mut webs);

    for w in &webs {
        assert!(
            !LUAU_KEYWORDS.contains(&w.assigned_name.as_str()),
            "Assigned name '{}' must not be a bare Luau keyword",
            w.assigned_name
        );
    }
}

// ============================================================================
// 4. LINEAR SCAN INTERVAL COLORING & DENSE OVERLAPS STRESS TEST
// ============================================================================

#[test]
fn test_adv_namer_linear_scan_dense_overlapping_stress_5000_intervals() {
    // Generate 5,000 intervals that overlap densely:
    // Each interval spans [i * 2, i * 2 + 200]
    // At any point, there are up to 100 concurrent active intervals.
    let num_intervals = 5000;
    let mut webs = Vec::with_capacity(num_intervals);

    for i in 0..num_intervals {
        let mut web = VariableWeb::new(WebId(i as u32), (i % 64) as u16);
        web.live_interval = (i * 2, i * 2 + 200);
        webs.push(web);
    }

    let namer = IntervalNamer::new(NamerConfig {
        reg_prefix: "v".to_string(),
        enable_semantic_inference: false,
        max_local_budget: 180,
    });

    let start = Instant::now();
    namer.assign_names(&mut webs);
    let elapsed = start.elapsed();

    // Verify O(N log N) scaling: 5,000 intervals must complete well under 100ms
    assert!(
        elapsed.as_millis() < 100,
        "5,000 interval linear scan coloring took {:?} (expected < 100ms)",
        elapsed
    );

    // Verify zero collision: sample pairs of overlapping intervals
    // For every step, check active set at that instant
    let mut active: HashMap<String, usize> = HashMap::new(); // name -> end_pc
    let mut order: Vec<usize> = (0..webs.len()).collect();
    order.sort_unstable_by_key(|&idx| webs[idx].live_interval.0);

    for &idx in &order {
        let start_pc = webs[idx].live_interval.0;
        let end_pc = webs[idx].live_interval.1;
        let name = &webs[idx].assigned_name;

        // Expire finished intervals
        active.retain(|_, &mut active_end| active_end >= start_pc);

        // Check for collision
        assert!(
            !active.contains_key(name),
            "COLLISION DETECTED: Variable '{}' assigned to interval [{}, {}] while already active until {}",
            name,
            start_pc,
            end_pc,
            active[name]
        );

        active.insert(name.clone(), end_pc);
    }

    // Verify colors are recycled: max concurrent active was ~101, so total distinct names should be ~101
    let distinct_names: HashSet<&str> = webs.iter().map(|w| w.assigned_name.as_str()).collect();
    assert!(
        distinct_names.len() <= 120,
        "Expected ~101 recycled colors, but found {} distinct names",
        distinct_names.len()
    );
}

#[test]
fn test_adv_namer_scope_budget_stress_300_locals_deep_nesting() {
    // Build 300 sequential local assignments
    let mut stmts = Vec::new();
    for i in 0..300 {
        let name: &'static str = Box::leak(format!("var_{}", i).into_boxed_str());
        stmts.push(LuauStat::LocalAssign {
            names: vec![name],
            values: vec![LuauExpr::Number(i as f64)],
        });
    }

    let budget = 180;
    let fitted = limit_locals(stmts, 0, budget);

    // Verify that every flat segment or DoBlock has local count <= budget
    fn verify_budget<'arena>(stmts: &[LuauStat<'arena>], outer: usize, budget: usize) {
        let flat_decls: usize = stmts.iter().map(|s| declared_here(s).len()).sum();
        assert!(
            outer + flat_decls <= budget,
            "Scope violation: outer ({}) + flat declarations ({}) = {} > budget ({})",
            outer,
            flat_decls,
            outer + flat_decls,
            budget
        );

        for s in stmts {
            match s {
                LuauStat::DoBlock(body) => {
                    verify_budget(body, outer, budget);
                }
                LuauStat::If { then_branch, else_branch, .. } => {
                    verify_budget(then_branch, outer, budget);
                    if let Some(els) = else_branch {
                        verify_budget(els, outer, budget);
                    }
                }
                _ => {}
            }
        }
    }

    verify_budget(&fitted, 0, budget);
}
