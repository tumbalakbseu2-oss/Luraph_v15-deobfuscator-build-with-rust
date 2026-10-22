//! Comprehensive unit tests for SSA variable web coalescence, reaching definitions,
//! and dead store elimination.

use bumpalo::Bump;
use luraph_deobf_rs::cfg::ir::{BasicBlock, BlockId, ControlFlowGraph, IrExpr, IrInst, Terminator};
use luraph_deobf_rs::ssa::{
    analyze_ssa, eliminate_dead_stores, run_reaching_definitions,
    DefKey, DefSite, UseSite,
};

#[test]
fn test_reaching_defs_linear_chain() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0: R(0) = 10; R(1) = R(0) + 1; Goto(1)
    let mut b0 = BasicBlock::new(BlockId(0), (0, 1), Terminator::Goto(BlockId(1)));
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(10.0),
    });
    let r0_use = bump.alloc(IrExpr::Register(0));
    let one = bump.alloc(IrExpr::Number(1.0));
    b0.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Binary {
            op: luraph_deobf_rs::cfg::BinOp::Add,
            left: r0_use,
            right: one,
        },
    });

    // Block 1: Return R(1)
    let r1_use: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(1)]);
    let b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Return(r1_use));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.recompute_successors();

    let dataflow = run_reaching_definitions(&cfg, 0);

    // Verify block 1 IN state has R(0) defined at stmt(0, 0) and R(1) defined at stmt(0, 1)
    let b1_in = &dataflow.in_states[&BlockId(1)];
    let r0_defs = &b1_in[&DefKey::Reg(0)];
    assert_eq!(r0_defs.len(), 1);
    assert!(r0_defs.contains(&DefSite::inst(BlockId(0), 0, 0)));

    let r1_defs = &b1_in[&DefKey::Reg(1)];
    assert_eq!(r1_defs.len(), 1);
    assert!(r1_defs.contains(&DefSite::inst(BlockId(0), 1, 1)));
}

#[test]
fn test_reaching_defs_diamond_branch() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0: branch to 1 or 2
    let b0 = BasicBlock::new(
        BlockId(0),
        (0, 0),
        Terminator::Branch {
            cond: IrExpr::Boolean(true),
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );

    // Block 1: R(0) = 100; goto 3
    let mut b1 = BasicBlock::new(BlockId(1), (1, 1), Terminator::Goto(BlockId(3)));
    b1.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(100.0),
    });

    // Block 2: R(0) = 200; goto 3
    let mut b2 = BasicBlock::new(BlockId(2), (2, 2), Terminator::Goto(BlockId(3)));
    b2.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(200.0),
    });

    // Block 3: Return R(0)
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
    let b3 = BasicBlock::new(BlockId(3), (3, 3), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    let dataflow = run_reaching_definitions(&cfg, 0);

    // Block 3 IN state must contain BOTH definition sites for R(0)
    let b3_in = &dataflow.in_states[&BlockId(3)];
    let r0_defs = &b3_in[&DefKey::Reg(0)];
    assert_eq!(r0_defs.len(), 2, "Both diamond branch definitions must reach Block 3");
    assert!(r0_defs.contains(&DefSite::inst(BlockId(1), 0, 0)));
    assert!(r0_defs.contains(&DefSite::inst(BlockId(2), 0, 0)));
}

#[test]
fn test_disjoint_set_union_find_coalescence() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Diamond CFG joining at Block 3: uses R(0)
    let b0 = BasicBlock::new(
        BlockId(0),
        (0, 0),
        Terminator::Branch {
            cond: IrExpr::Boolean(true),
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );

    let mut b1 = BasicBlock::new(BlockId(1), (1, 1), Terminator::Goto(BlockId(3)));
    b1.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(10.0),
    });

    let mut b2 = BasicBlock::new(BlockId(2), (2, 2), Terminator::Goto(BlockId(3)));
    b2.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(20.0),
    });

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
    let b3 = BasicBlock::new(BlockId(3), (3, 3), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);

    // Both def sites must be coalesced into the exact same WebId!
    let d1 = DefSite::inst(BlockId(1), 0, 0);
    let d2 = DefSite::inst(BlockId(2), 0, 0);

    let w1 = ssa.def_to_web.get(&d1).expect("d1 mapped");
    let w2 = ssa.def_to_web.get(&d2).expect("d2 mapped");
    assert_eq!(w1, w2, "Both reaching definitions must be coalesced into identical VariableWeb");

    // The use at Block 3 terminator must point to that same WebId
    let u3 = UseSite::term(BlockId(3), 0);
    assert_eq!(ssa.use_to_web.get(&u3), Some(&Some(*w1)));
}

#[test]
fn test_upvalue_mutational_sharing() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // 0: R(0) = 42
    // 1: R(1) = Closure(upvalues: [R(0)]) -> Captures R(0)
    // 2: R(0) = 99 -> Mutates captured R(0) before close!
    // 3: Return
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (0, 3),
        Terminator::Return(bump.alloc_slice_clone(&[])),
    );
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(42.0),
    });

    let upvals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
    b0.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Closure {
            proto_id: 1,
            upvalues: upvals,
        },
    });

    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(99.0),
    });

    cfg.add_block(b0);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);

    let d0 = DefSite::inst(BlockId(0), 0, 0);
    let d2 = DefSite::inst(BlockId(0), 2, 0);

    let w0 = ssa.def_to_web.get(&d0).expect("d0 mapped");
    let w2 = ssa.def_to_web.get(&d2).expect("d2 mapped");

    // Mutation must be coalesced with the captured variable web
    assert_eq!(w0, w2, "Redefinition of open upvalue must join the captured variable web");
    let web = ssa.get_web(*w0).unwrap();
    assert!(web.is_upvalue, "Coalesced web must be marked as upvalue");
}

#[test]
fn test_upvalue_close_partitioning() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // 0: R(0) = 1
    // 1: R(1) = Closure(upvalues: [R(0)])
    // 2: CloseUpvalue(0)
    // 3: R(0) = 2  -> Occurs AFTER close! Must get a fresh web!
    // 4: Return R(0)
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
    let mut b0 = BasicBlock::new(BlockId(0), (0, 4), Terminator::Return(ret_vals));
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(1.0),
    });

    let upvals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
    b0.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Closure {
            proto_id: 1,
            upvalues: upvals,
        },
    });

    b0.instructions.push(IrInst::CloseUpvalue(0));

    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(2.0),
    });

    cfg.add_block(b0);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);

    let d_before = DefSite::inst(BlockId(0), 0, 0);
    let d_after = DefSite::inst(BlockId(0), 3, 0);

    let w_before = ssa.def_to_web.get(&d_before).expect("before mapped");
    let w_after = ssa.def_to_web.get(&d_after).expect("after mapped");

    assert_ne!(
        w_before, w_after,
        "Definition after CloseUpvalue must receive a distinct, non-aliased WebId"
    );
    assert!(ssa.get_web(*w_before).unwrap().is_upvalue);
}

#[test]
fn test_loop_variable_web_isolation() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Numeric For Loop Header: var = 5
    let b0 = BasicBlock::new(
        BlockId(0),
        (0, 0),
        Terminator::ForLoop {
            var: 5,
            start: IrExpr::Number(1.0),
            limit: IrExpr::Number(10.0),
            step: Some(IrExpr::Number(1.0)),
            body: BlockId(1),
            exit: BlockId(2),
        },
    );

    // Body: R(6) = R(5) + 1; Goto(0)
    let mut b1 = BasicBlock::new(BlockId(1), (1, 1), Terminator::Goto(BlockId(0)));
    let r5 = bump.alloc(IrExpr::Register(5));
    let one = bump.alloc(IrExpr::Number(1.0));
    b1.instructions.push(IrInst::Assign {
        dest: 6,
        src: IrExpr::Binary {
            op: luraph_deobf_rs::cfg::BinOp::Add,
            left: r5,
            right: one,
        },
    });

    // Exit: Return
    let b2 = BasicBlock::new(
        BlockId(2),
        (2, 2),
        Terminator::Return(bump.alloc_slice_clone(&[])),
    );

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);

    let loop_def = DefSite::term(BlockId(0), 5);
    let web_id = ssa.def_to_web.get(&loop_def).expect("loop def mapped");
    assert!(ssa.loop_webs.contains(web_id), "Loop variable must be tracked in loop_webs");
    let web = ssa.get_web(*web_id).unwrap();
    assert!(web.is_loop_var, "Loop variable web must have is_loop_var set to true");
}

#[test]
fn test_dead_store_elimination_pure_constant() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // 0: R(0) = 999 (never used!)
    // 1: R(1) = 42
    // 2: Return R(1)
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (0, 2),
        Terminator::Return(bump.alloc_slice_clone(&[IrExpr::Register(1)])),
    );
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(999.0),
    });
    b0.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(42.0),
    });

    cfg.add_block(b0);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);
    let eliminated = eliminate_dead_stores(&mut cfg, &ssa);

    assert_eq!(eliminated, 1, "Exactly one dead store should be eliminated");
    let block = &cfg.blocks[&BlockId::ENTRY];
    assert!(matches!(block.instructions[0], IrInst::Nop), "Dead store must be rewritten to Nop");
    assert!(matches!(block.instructions[1], IrInst::Assign { dest: 1, .. }));
}

#[test]
fn test_dead_store_preserves_side_effects() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // 0: R(0) = print("hello") (Call has side-effects, must NOT be eliminated!)
    // 1: Return
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (0, 1),
        Terminator::Return(bump.alloc_slice_clone(&[])),
    );

    let callee = bump.alloc(IrExpr::Global("print"));
    let args: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::String("hello")]);
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Call { callee, args },
    });

    cfg.add_block(b0);
    cfg.recompute_successors();

    let ssa = analyze_ssa(&cfg, 0);
    let eliminated = eliminate_dead_stores(&mut cfg, &ssa);

    assert_eq!(eliminated, 0, "Side-effecting call assignment must NOT be eliminated as dead store");
    let block = &cfg.blocks[&BlockId::ENTRY];
    assert!(matches!(block.instructions[0], IrInst::Assign { dest: 0, .. }));
}
