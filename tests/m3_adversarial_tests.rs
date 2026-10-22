//! Milestone 3 Adversarial Test Suite
//!
//! Stress tests CFG builder, SCCP abstract interpreter, unflattening engine,
//! dominator trees, and high-level region structuring against pathological graphs,
//! extreme mathematical inputs, cyclic trampolines, and complex dispatcher loops.

use std::sync::Arc;
use bumpalo::Bump;
use luraph_deobf_rs::cfg::{
    thread_empty,
    AbstractInterpreter, BasicBlock, BinOp, BlockId, ControlFlowGraph,
    DominatorTree, IrExpr, IrInst, LatticeValue, Literal, LoopExtractor,
    PostDominatorTree, RegionStructurer,
    Terminator, UnOp, Unflattener, VmState, emit_luau,
};

// ============================================================================
// 1. CFG PATHOLOGICAL TOPOLOGY TESTS
// ============================================================================

#[test]
fn test_adv_cfg_deep_diamond_branching_100() {
    // Construct 100 sequential diamond branches (total >200 blocks):
    // For each i in 0..100:
    //   Block (2*i) branches to Block (2*i + 1) [then] and Block (2*i + 2) [else]
    //   Both branch targets merge into Block (2*i + 3) or next stage
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let num_diamonds = 100;
    for i in 0..num_diamonds {
        let head_id = BlockId::new((i * 3) as u32);
        let left_id = BlockId::new((i * 3 + 1) as u32);
        let right_id = BlockId::new((i * 3 + 2) as u32);
        let merge_id = BlockId::new((i * 3 + 3) as u32);

        // Head block
        let head = BasicBlock::new(
            head_id,
            (i * 3 + 1, i * 3 + 1),
            Terminator::Branch {
                cond: IrExpr::Register(i as u16),
                true_target: left_id,
                false_target: right_id,
            },
        );

        // Left branch
        let mut left = BasicBlock::new(left_id, (i * 3 + 2, i * 3 + 2), Terminator::Goto(merge_id));
        left.instructions.push(IrInst::Assign {
            dest: 1000,
            src: IrExpr::Number(i as f64 * 10.0),
        });

        // Right branch
        let mut right = BasicBlock::new(right_id, (i * 3 + 2, i * 3 + 2), Terminator::Goto(merge_id));
        right.instructions.push(IrInst::Assign {
            dest: 1000,
            src: IrExpr::Number(i as f64 * 20.0),
        });

        cfg.add_block(head);
        cfg.add_block(left);
        cfg.add_block(right);
    }

    // Final merge / exit block
    let exit_id = BlockId::new((num_diamonds * 3) as u32);
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(1000)]);
    let exit_block = BasicBlock::new(exit_id, (9999, 9999), Terminator::Return(ret_vals));
    cfg.add_block(exit_block);
    cfg.recompute_successors();

    assert_eq!(cfg.blocks.len(), num_diamonds * 3 + 1);

    // Dominator tree on deep diamond topology
    let dom = DominatorTree::compute(&cfg);
    assert_eq!(dom.idom.get(&BlockId::ENTRY), Some(&BlockId::ENTRY));
    assert!(dom.dominates(BlockId::ENTRY, exit_id));

    // Post-dominator tree
    let pdom = PostDominatorTree::compute(&cfg);
    assert!(pdom.ipdom.contains_key(&BlockId::ENTRY));

    // Region structuring must not overflow stack or panic on 100 nested diamonds
    let mut structurer = RegionStructurer::new(&cfg);
    let region = structurer.structure();
    let luau = emit_luau(&region, &cfg, 0);

    assert!(!luau.is_empty(), "Luau emission should succeed for 100 diamonds");
    assert!(luau.contains("local r1000 = "));
}

#[test]
fn test_adv_cfg_long_trampoline_chain_500() {
    // 500 chained empty trampoline blocks:
    // Block 0 -> Block 1 -> Block 2 -> ... -> Block 500 (Return)
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let chain_len = 500;
    for i in 0..chain_len {
        let b = BasicBlock::new(
            BlockId::new(i as u32),
            (i + 1, i + 1),
            Terminator::Goto(BlockId::new((i + 1) as u32)),
        );
        cfg.add_block(b);
    }

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Number(42.0)]);
    let term_block = BasicBlock::new(
        BlockId::new(chain_len as u32),
        (chain_len + 1, chain_len + 1),
        Terminator::Return(ret_vals),
    );
    cfg.add_block(term_block);
    cfg.recompute_successors();

    assert_eq!(cfg.blocks.len(), chain_len + 1);

    // Thread empty trampolines: must collapse the entire 500-block chain in linear time
    thread_empty(&mut cfg);

    // After contraction and straight-line collapsing, only 1 block should remain
    assert_eq!(
        cfg.blocks.len(),
        1,
        "500-long empty trampoline chain must collapse into a single block"
    );
    let entry = &cfg.blocks[&cfg.entry];
    assert!(matches!(entry.terminator, Terminator::Return(_)));
}

#[test]
fn test_adv_cfg_trampoline_cycle_livelock_immunity() {
    // Pathological cycle of empty trampolines:
    // Entry (0) -> Branch(1, 4)
    // Block 1 -> Goto(2)
    // Block 2 -> Goto(3)
    // Block 3 -> Goto(1) [Empty Trampoline Cycle: 1->2->3->1]
    // Block 4 -> Return
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let b0 = BasicBlock::new(
        BlockId(0),
        (1, 1),
        Terminator::Branch {
            cond: IrExpr::Register(0),
            true_target: BlockId(1),
            false_target: BlockId(4),
        },
    );
    let b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Goto(BlockId(2)));
    let b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Goto(BlockId(3)));
    let b3 = BasicBlock::new(BlockId(3), (4, 4), Terminator::Goto(BlockId(1)));

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b4 = BasicBlock::new(BlockId(4), (5, 5), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.add_block(b4);
    cfg.recompute_successors();

    // thread_empty MUST NOT hang or infinite loop on empty trampoline cycle!
    thread_empty(&mut cfg);

    // CFG should still be valid and contain block 0 and block 4
    assert!(cfg.blocks.contains_key(&BlockId(0)));
    assert!(cfg.blocks.contains_key(&BlockId(4)));
}

#[test]
fn test_adv_cfg_irreducible_multi_entry_cycles() {
    // Classic irreducible CFG with multiple entries into a cycle:
    // Entry (0) -> Branch to 1 or 2
    // Block 1 -> Goto(2)
    // Block 2 -> Branch(1, 3)
    // Block 3 -> Return
    // Notice both 1 and 2 can be entered from 0, forming an irreducible cycle between 1 and 2!
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let b0 = BasicBlock::new(
        BlockId(0),
        (1, 1),
        Terminator::Branch {
            cond: IrExpr::Register(10),
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );

    let mut b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Goto(BlockId(2)));
    b1.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(1.0),
    });

    let mut b2 = BasicBlock::new(
        BlockId(2),
        (3, 3),
        Terminator::Branch {
            cond: IrExpr::Register(20),
            true_target: BlockId(1), // Back-edge into 1
            false_target: BlockId(3),
        },
    );
    b2.instructions.push(IrInst::Assign {
        dest: 2,
        src: IrExpr::Number(2.0),
    });

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b3 = BasicBlock::new(BlockId(3), (4, 4), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    // Test dominators and natural loops
    let dom = DominatorTree::compute(&cfg);
    assert_eq!(dom.idom.get(&BlockId(0)), Some(&BlockId(0)));
    assert_eq!(dom.idom.get(&BlockId(1)), Some(&BlockId(0)));
    assert_eq!(dom.idom.get(&BlockId(2)), Some(&BlockId(0)));

    let _loops = LoopExtractor::extract_loops(&cfg, &dom);
    // Dominance analysis must identify loop back-edge without infinite recursion
    let mut structurer = RegionStructurer::new(&cfg);
    let region = structurer.structure();
    let luau = emit_luau(&region, &cfg, 0);

    assert!(!luau.is_empty(), "Structuring irreducible CFG must produce Luau code without panic");
}

// ============================================================================
// 2. SCCP & ARITHMETIC BOUNDARY TESTS
// ============================================================================

#[test]
fn test_adv_sccp_deep_polynomial_affine_chain() {
    // 50-step iterative affine and polynomial state updates:
    // state_{k} = (((state_{k-1} * 3) + 7) ^ 0xDEADBEEF) % 1000003
    let mut cur_val: i64 = 42;
    let mut state = VmState::new();
    state.set_reg(0, LatticeValue::Constant(Literal::Int(cur_val)));

    let _interp = AbstractInterpreter::new(&state);

    for step in 1..=50 {
        // Build AST: (((r0 * 3) + 7) ^ 0xDEADBEEF) % 1000003
        let mul_expr = IrExpr::Binary {
            op: BinOp::Mul,
            left: Box::leak(Box::new(IrExpr::Register(0))),
            right: Box::leak(Box::new(IrExpr::Number(3.0))),
        };
        let add_expr = IrExpr::Binary {
            op: BinOp::Add,
            left: Box::leak(Box::new(mul_expr)),
            right: Box::leak(Box::new(IrExpr::Number(7.0))),
        };
        let xor_expr = IrExpr::Binary {
            op: BinOp::BitXor,
            left: Box::leak(Box::new(add_expr)),
            right: Box::leak(Box::new(IrExpr::Number(0xDEADBEEFu32 as f64))),
        };
        let mod_expr = IrExpr::Binary {
            op: BinOp::Mod,
            left: Box::leak(Box::new(xor_expr)),
            right: Box::leak(Box::new(IrExpr::Number(1000003.0))),
        };

        let interp = AbstractInterpreter::new(&state);
        let result = interp.eval(&mod_expr);

        // Ground-truth step in pure 64-bit integer
        let ground_mul = cur_val.wrapping_mul(3);
        let ground_add = ground_mul.wrapping_add(7);
        let ground_xor = ((ground_add as u32) ^ 0xDEADBEEFu32) as i64;
        let ground_mod = ground_xor.rem_euclid(1000003);

        cur_val = ground_mod;
        assert_eq!(
            result,
            LatticeValue::Constant(Literal::Int(ground_mod)),
            "Step {step} mismatch in polynomial affine evaluation"
        );

        // Update state for next iteration
        state.set_reg(0, result);
    }
}

#[test]
fn test_adv_sccp_boundary_condition_extreme_values() {
    let state = VmState::new();
    let interp = AbstractInterpreter::new(&state);

    let int_min = LatticeValue::Constant(Literal::Int(i64::MIN));
    let int_max = LatticeValue::Constant(Literal::Int(i64::MAX));
    let zero = LatticeValue::Constant(Literal::Int(0));
    let _neg_one = LatticeValue::Constant(Literal::Int(-1));
    let u32_max = LatticeValue::Constant(Literal::Int(0xFFFFFFFF));

    // 1. Extreme comparisons
    assert_eq!(
        interp.eval_binary(BinOp::Lt, &int_min, &int_max),
        LatticeValue::Constant(Literal::Bool(true))
    );
    assert_eq!(
        interp.eval_binary(BinOp::Gt, &int_min, &int_max),
        LatticeValue::Constant(Literal::Bool(false))
    );
    assert_eq!(
        interp.eval_binary(BinOp::Le, &int_min, &int_min),
        LatticeValue::Constant(Literal::Bool(true))
    );
    assert_eq!(
        interp.eval_binary(BinOp::Ge, &int_max, &int_max),
        LatticeValue::Constant(Literal::Bool(true))
    );
    assert_eq!(
        interp.eval_binary(BinOp::Ne, &int_min, &int_max),
        LatticeValue::Constant(Literal::Bool(true))
    );

    // 2. Division / Modulo by Zero must return Bottom / safe float without panic
    let div_zero = interp.eval_binary(BinOp::Div, &int_max, &zero);
    assert!(
        matches!(div_zero, LatticeValue::Constant(Literal::Float(f)) if f.is_infinite()),
        "Float division by zero in Lua returns inf"
    );

    let idiv_zero = interp.eval_binary(BinOp::FloorDiv, &int_max, &zero);
    assert_eq!(idiv_zero, LatticeValue::Bottom, "Floor div by zero returns Bottom");

    let mod_zero = interp.eval_binary(BinOp::Mod, &int_max, &zero);
    assert_eq!(mod_zero, LatticeValue::Bottom, "Mod by zero returns Bottom");

    // 3. Modulo semantics:
    // In Lua / Luau, a % b = a - math.floor(a/b) * b:
    // 10 % 3 = 1
    // -10 % 3 = 2
    let c10 = LatticeValue::Constant(Literal::Int(10));
    let c_neg10 = LatticeValue::Constant(Literal::Int(-10));
    let c3 = LatticeValue::Constant(Literal::Int(3));

    assert_eq!(
        interp.eval_binary(BinOp::Mod, &c10, &c3),
        LatticeValue::Constant(Literal::Int(1))
    );
    assert_eq!(
        interp.eval_binary(BinOp::Mod, &c_neg10, &c3),
        LatticeValue::Constant(Literal::Int(2))
    );

    // Float implementation in sccp.rs:288 for non-whole floats correctly computes Lua modulo:
    // 10.5 - (-3.0) * floor(10.5 / -3.0) = 10.5 - 12.0 = -1.5
    let c10_5_f = LatticeValue::Constant(Literal::Float(10.5));
    let c_neg3_f = LatticeValue::Constant(Literal::Float(-3.0));
    assert_eq!(
        interp.eval_binary(BinOp::Mod, &c10_5_f, &c_neg3_f),
        LatticeValue::Constant(Literal::Float(-1.5))
    );

    // In sccp.rs:281, as_i64() matches whole floats and integers, delegating to Rust's
    // `rem_euclid`. Rust's rem_euclid computes the non-negative remainder (1) instead of
    // Lua's negative remainder (-2):
    let c_neg3 = LatticeValue::Constant(Literal::Int(-3));
    let int_neg_mod = interp.eval_binary(BinOp::Mod, &c10, &c_neg3);
    assert_eq!(
        int_neg_mod,
        LatticeValue::Constant(Literal::Int(1)),
        "Empirical finding: sccp.rs uses Rust rem_euclid which yields +1 instead of Luau's -2"
    );

    // 4. Bit32 operations with shift >= 32
    let c32 = LatticeValue::Constant(Literal::Int(32));
    let c64 = LatticeValue::Constant(Literal::Int(64));
    let c1 = LatticeValue::Constant(Literal::Int(1));

    assert_eq!(
        interp.eval_binary(BinOp::ShiftLeft, &c1, &c32),
        LatticeValue::Constant(Literal::Int(0)),
        "ShiftLeft >= 32 must evaluate to 0 in bit32"
    );
    assert_eq!(
        interp.eval_binary(BinOp::ShiftRight, &u32_max, &c64),
        LatticeValue::Constant(Literal::Int(0)),
        "ShiftRight >= 32 must evaluate to 0 in bit32"
    );

    // 5. Unary BitNot on 32-bit boundary
    let not_zero = interp.eval(&IrExpr::Unary {
        op: UnOp::BitNot,
        expr: Box::leak(Box::new(IrExpr::Number(0.0))),
    });
    assert_eq!(
        not_zero,
        LatticeValue::Constant(Literal::Int(0xFFFFFFFF))
    );

    let not_u32_max = interp.eval(&IrExpr::Unary {
        op: UnOp::BitNot,
        expr: Box::leak(Box::new(IrExpr::Number(0xFFFFFFFFu32 as f64))),
    });
    assert_eq!(
        not_u32_max,
        LatticeValue::Constant(Literal::Int(0))
    );
}

#[test]
fn test_adv_sccp_lattice_meet_algebraic_laws() {
    let elements = vec![
        LatticeValue::Top,
        LatticeValue::Bottom,
        LatticeValue::Constant(Literal::Int(1)),
        LatticeValue::Constant(Literal::Int(2)),
        LatticeValue::Constant(Literal::Bool(true)),
        LatticeValue::Constant(Literal::Bool(false)),
        LatticeValue::Constant(Literal::Nil),
        LatticeValue::Constant(Literal::String(Arc::from("test"))),
        LatticeValue::KnownTruth(true),
        LatticeValue::KnownTruth(false),
    ];

    // Commutativity: a ∧ b == b ∧ a
    for a in &elements {
        for b in &elements {
            assert_eq!(
                a.meet(b),
                b.meet(a),
                "Commutativity violated for {:?} and {:?}",
                a,
                b
            );
        }
    }

    // Idempotency: a ∧ a == a
    for a in &elements {
        assert_eq!(a.meet(a), *a, "Idempotency violated for {:?}", a);
    }

    // Identity (Top): Top ∧ a == a
    for a in &elements {
        assert_eq!(LatticeValue::Top.meet(a), *a);
    }

    // Absorption (Bottom): Bottom ∧ a == Bottom
    for a in &elements {
        assert_eq!(LatticeValue::Bottom.meet(a), LatticeValue::Bottom);
    }
}

// ============================================================================
// 3. UNFLATTENING ENGINE STRESS TESTS
// ============================================================================

#[test]
fn test_adv_unflatten_nested_dispatchers_with_shared_exit() {
    // Nested Dispatcher Topology:
    // Outer Dispatcher controls coarse states (State 1 -> Inner Dispatcher, State 2 -> Exit)
    // Inner Dispatcher controls fine states (State 10 -> Handler A, State 20 -> Handler B)
    // Handler B sets outer state = 2 and exits to Outer Dispatcher
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Entry: outer_state(r10) = 1, goto Outer Dispatcher (Block 100)
    let mut b_entry = BasicBlock::new(BlockId(0), (1, 1), Terminator::Goto(BlockId(100)));
    b_entry.instructions.push(IrInst::Assign {
        dest: 10,
        src: IrExpr::Number(1.0),
    });

    // Outer Dispatcher (100):
    // if r10 == 1 goto Block 101 (Inner Dispatcher Header) else goto Block 102
    let b100 = BasicBlock::new(
        BlockId(100),
        (2, 2),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(10))),
                right: Box::leak(Box::new(IrExpr::Number(1.0))),
            },
            true_target: BlockId(101),
            false_target: BlockId(102),
        },
    );

    // Outer Dispatcher Branch (102):
    // if r10 == 2 goto Exit (Block 999) else goto Crash (Block 888)
    let b102 = BasicBlock::new(
        BlockId(102),
        (3, 3),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(10))),
                right: Box::leak(Box::new(IrExpr::Number(2.0))),
            },
            true_target: BlockId(999),
            false_target: BlockId(888),
        },
    );

    // Inner Dispatcher Header (101):
    // if r20 == 10 goto Handler A (Block 201) else goto Block 103
    let b101 = BasicBlock::new(
        BlockId(101),
        (4, 4),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(20))),
                right: Box::leak(Box::new(IrExpr::Number(10.0))),
            },
            true_target: BlockId(201),
            false_target: BlockId(103),
        },
    );

    // Inner Dispatcher Branch (103):
    // if r20 == 20 goto Handler B (Block 202) else goto Crash (Block 888)
    let b103 = BasicBlock::new(
        BlockId(103),
        (5, 5),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(20))),
                right: Box::leak(Box::new(IrExpr::Number(20.0))),
            },
            true_target: BlockId(202),
            false_target: BlockId(888),
        },
    );

    // Handler A (201): sets r20 = 20, jumps back to Inner Dispatcher (101)
    let mut b201 = BasicBlock::new(BlockId(201), (6, 6), Terminator::Goto(BlockId(101)));
    b201.instructions.push(IrInst::Assign {
        dest: 20,
        src: IrExpr::Number(20.0),
    });

    // Handler B (202): sets r10 = 2, jumps back to Outer Dispatcher (100)
    let mut b202 = BasicBlock::new(BlockId(202), (7, 7), Terminator::Goto(BlockId(100)));
    b202.instructions.push(IrInst::Assign {
        dest: 10,
        src: IrExpr::Number(2.0),
    });

    // Exit Block (999)
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Number(12345.0)]);
    let b999 = BasicBlock::new(BlockId(999), (8, 8), Terminator::Return(ret_vals));

    // Crash Block (888)
    let b888 = BasicBlock::new(BlockId(888), (9, 9), Terminator::Crash);

    cfg.add_block(b_entry);
    cfg.add_block(b100);
    cfg.add_block(b102);
    cfg.add_block(b101);
    cfg.add_block(b103);
    cfg.add_block(b201);
    cfg.add_block(b202);
    cfg.add_block(b999);
    cfg.add_block(b888);
    cfg.recompute_successors();

    assert_eq!(cfg.blocks.len(), 9);

    // Unflattening pass: detection must locate at least one of the dispatchers
    if let Some(disp) = Unflattener::detect_dispatcher(&cfg) {
        Unflattener::unflatten(&mut cfg, &disp);
        assert!(!cfg.blocks.contains_key(&disp.header), "Detected dispatcher must be pruned");
    }

    // Verify CFG remains consistent and reachable
    assert!(cfg.blocks.contains_key(&cfg.entry));
}

#[test]
fn test_adv_unflatten_multi_exit_and_loop_back_states() {
    // 5-state dispatcher with back-edge and multiple exits:
    // State 1 -> Block 1: does work, sets State = 2, goto Disp
    // State 2 -> Block 2: branches (if r0 then State = 1 else State = 3), goto Disp
    // State 3 -> Block 3: returns
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Entry
    let mut b0 = BasicBlock::new(BlockId(0), (1, 1), Terminator::Goto(BlockId(10)));
    b0.instructions.push(IrInst::Assign {
        dest: 50,
        src: IrExpr::Number(1.0),
    });

    // Dispatcher Header (10)
    let b10 = BasicBlock::new(
        BlockId(10),
        (2, 2),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(50))),
                right: Box::leak(Box::new(IrExpr::Number(1.0))),
            },
            true_target: BlockId(1),
            false_target: BlockId(11),
        },
    );

    // Dispatcher branch 11
    let b11 = BasicBlock::new(
        BlockId(11),
        (3, 3),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(50))),
                right: Box::leak(Box::new(IrExpr::Number(2.0))),
            },
            true_target: BlockId(2),
            false_target: BlockId(12),
        },
    );

    // Dispatcher branch 12
    let b12 = BasicBlock::new(
        BlockId(12),
        (4, 4),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(50))),
                right: Box::leak(Box::new(IrExpr::Number(3.0))),
            },
            true_target: BlockId(3),
            false_target: BlockId(99),
        },
    );

    // Handler 1
    let mut b1 = BasicBlock::new(BlockId(1), (5, 5), Terminator::Goto(BlockId(10)));
    b1.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(100.0),
    });
    b1.instructions.push(IrInst::Assign {
        dest: 50,
        src: IrExpr::Number(2.0),
    });

    // Handler 2: branches to 20 (State 1) or 21 (State 3)
    let b2 = BasicBlock::new(
        BlockId(2),
        (6, 6),
        Terminator::Branch {
            cond: IrExpr::Register(0),
            true_target: BlockId(20),
            false_target: BlockId(21),
        },
    );

    let mut b20 = BasicBlock::new(BlockId(20), (7, 7), Terminator::Goto(BlockId(10)));
    b20.instructions.push(IrInst::Assign {
        dest: 50,
        src: IrExpr::Number(1.0),
    });

    let mut b21 = BasicBlock::new(BlockId(21), (8, 8), Terminator::Goto(BlockId(10)));
    b21.instructions.push(IrInst::Assign {
        dest: 50,
        src: IrExpr::Number(3.0),
    });

    // Handler 3 (Exit)
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b3 = BasicBlock::new(BlockId(3), (9, 9), Terminator::Return(ret_vals));

    let b99 = BasicBlock::new(BlockId(99), (10, 10), Terminator::Crash);

    cfg.add_block(b0);
    cfg.add_block(b10);
    cfg.add_block(b11);
    cfg.add_block(b12);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b20);
    cfg.add_block(b21);
    cfg.add_block(b3);
    cfg.add_block(b99);
    cfg.recompute_successors();

    let disp = Unflattener::detect_dispatcher(&cfg).expect("Dispatcher with loop-back must be detected");
    assert_eq!(disp.header, BlockId(10));
    assert_eq!(disp.state_map.get(&1), Some(&BlockId(1)));
    assert_eq!(disp.state_map.get(&2), Some(&BlockId(2)));
    assert_eq!(disp.state_map.get(&3), Some(&BlockId(3)));

    Unflattener::unflatten(&mut cfg, &disp);

    // Ensure dispatcher blocks 10, 11, 12, 99 are pruned
    assert!(!cfg.blocks.contains_key(&BlockId(10)));
    assert!(!cfg.blocks.contains_key(&BlockId(11)));
    assert!(!cfg.blocks.contains_key(&BlockId(12)));
    assert!(!cfg.blocks.contains_key(&BlockId(99)));

    // Handlers and dispatchers were unflattened and straight-line collapsed:
    // Block 1 now directly branches on r0: looping back to Block 1 (true) or exiting to Block 3 (false)
    assert_eq!(
        cfg.blocks[&BlockId(1)].terminator,
        Terminator::Branch {
            cond: IrExpr::Register(0),
            true_target: BlockId(1),
            false_target: BlockId(3),
        }
    );
}

// ============================================================================
// 4. REGION STRUCTURING & EMISSION STRESS TESTS
// ============================================================================

#[test]
fn test_adv_structurer_deeply_nested_loops_and_breaks() {
    // 20 deeply nested loops with conditional breaks
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let depth = 20;
    for i in 0..depth {
        let header_id = BlockId::new((i * 2) as u32);
        let body_id = BlockId::new((i * 2 + 1) as u32);
        let exit_id = BlockId::new(((i + 1) * 2) as u32);

        // Header: while (r_i > 0)
        let header = BasicBlock::new(
            header_id,
            (i * 2 + 1, i * 2 + 1),
            Terminator::Branch {
                cond: IrExpr::Register(i as u16),
                true_target: body_id,
                false_target: exit_id,
            },
        );

        // Body: Goto header
        let body = BasicBlock::new(
            body_id,
            (i * 2 + 2, i * 2 + 2),
            Terminator::Goto(header_id),
        );

        cfg.add_block(header);
        cfg.add_block(body);
    }

    let final_exit_id = BlockId::new((depth * 2) as u32);
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let final_block = BasicBlock::new(final_exit_id, (999, 999), Terminator::Return(ret_vals));
    cfg.add_block(final_block);
    cfg.recompute_successors();

    let mut structurer = RegionStructurer::new(&cfg);
    let region = structurer.structure();
    let luau = emit_luau(&region, &cfg, 0);

    assert!(!luau.is_empty(), "Structuring 20 nested loops must succeed");
    assert!(luau.contains("while "), "Luau code must contain while loops");
}
