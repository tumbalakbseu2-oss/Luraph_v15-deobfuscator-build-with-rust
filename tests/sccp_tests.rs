//! Unit tests for Sparse Conditional Constant Propagation (SCCP),
//! 4-state Lattice operations, Luau arithmetic, bit32 evaluation, and opaque predicate pruning.

use std::sync::Arc;
use bumpalo::Bump;
use luraph_deobf_rs::cfg::{
    AbstractInterpreter, BasicBlock, BinOp, BlockId, ControlFlowGraph, IrExpr, IrInst,
    LatticeValue, Literal, SccpEngine, Terminator, UnOp, VmState,
};

#[test]
fn test_lattice_meet_axioms() {
    let top = LatticeValue::Top;
    let bot = LatticeValue::Bottom;
    let c1 = LatticeValue::Constant(Literal::Int(42));
    let c2 = LatticeValue::Constant(Literal::Int(42));
    let c3 = LatticeValue::Constant(Literal::Int(99));
    let c_falsy = LatticeValue::Constant(Literal::Bool(false));
    let kt_true = LatticeValue::KnownTruth(true);
    let kt_false = LatticeValue::KnownTruth(false);

    // Identity: Top meet x = x
    assert_eq!(top.meet(&c1), c1);
    assert_eq!(c1.meet(&top), c1);
    assert_eq!(top.meet(&bot), bot);

    // Absorbing element: Bottom meet x = Bottom
    assert_eq!(bot.meet(&c1), bot);
    assert_eq!(c1.meet(&bot), bot);

    // Equal constants meet to same constant
    assert_eq!(c1.meet(&c2), c1);

    // Different constants with identical truthiness degrade to KnownTruth
    assert_eq!(c1.meet(&c3), kt_true);

    // Constants with conflicting truthiness degrade to Bottom
    assert_eq!(c1.meet(&c_falsy), bot);

    // Constant and KnownTruth
    assert_eq!(c1.meet(&kt_true), kt_true);
    assert_eq!(c1.meet(&kt_false), bot);

    // KnownTruth meet KnownTruth
    assert_eq!(kt_true.meet(&kt_true), kt_true);
    assert_eq!(kt_true.meet(&kt_false), bot);
}

#[test]
fn test_luau_arithmetic_modulo_semantics() {
    let state = VmState::new();
    let interp = AbstractInterpreter::new(&state);

    // Positive modulo: 17 % 5 = 2
    let c17 = LatticeValue::Constant(Literal::Int(17));
    let c5 = LatticeValue::Constant(Literal::Int(5));
    let res = interp.eval_binary(BinOp::Mod, &c17, &c5);
    assert_eq!(res, LatticeValue::Constant(Literal::Int(2)));

    // Negative modulo matching Lua Euclidean semantics: -1 % 5 = 4
    let c_neg1 = LatticeValue::Constant(Literal::Int(-1));
    let res_neg = interp.eval_binary(BinOp::Mod, &c_neg1, &c5);
    assert_eq!(res_neg, LatticeValue::Constant(Literal::Int(4)));
}

#[test]
fn test_bit32_bitwise_operations() {
    let state = VmState::new();
    let interp = AbstractInterpreter::new(&state);

    let a = LatticeValue::Constant(Literal::Int(0x00FF00FF));
    let b = LatticeValue::Constant(Literal::Int(0x0F0F0F0F));

    // band: 0x000F000F
    let band_res = interp.eval_binary(BinOp::BitAnd, &a, &b);
    assert_eq!(band_res, LatticeValue::Constant(Literal::Int(0x000F000F)));

    // bxor: 0x0FE00FE0
    let bxor_res = interp.eval_binary(BinOp::BitXor, &a, &b);
    assert_eq!(bxor_res, LatticeValue::Constant(Literal::Int(0x00FF00FF ^ 0x0F0F0F0F)));

    // shift left: 1 << 4 = 16
    let one = LatticeValue::Constant(Literal::Int(1));
    let four = LatticeValue::Constant(Literal::Int(4));
    let shift_res = interp.eval_binary(BinOp::ShiftLeft, &one, &four);
    assert_eq!(shift_res, LatticeValue::Constant(Literal::Int(16)));
}

#[test]
fn test_affine_state_update_evaluation() {
    let mut state = VmState::new();
    state.set_reg(0, LatticeValue::Constant(Literal::Int(12345)));

    let interp = AbstractInterpreter::new(&state);

    // Formula: (r0 * 1664525 + 1013904223) % 1000003
    let left = IrExpr::Binary {
        op: BinOp::Mul,
        left: Box::leak(Box::new(IrExpr::Register(0))),
        right: Box::leak(Box::new(IrExpr::Number(1664525.0))),
    };
    let add_expr = IrExpr::Binary {
        op: BinOp::Add,
        left: Box::leak(Box::new(left)),
        right: Box::leak(Box::new(IrExpr::Number(1013904223.0))),
    };
    let mod_expr = IrExpr::Binary {
        op: BinOp::Mod,
        left: Box::leak(Box::new(add_expr)),
        right: Box::leak(Box::new(IrExpr::Number(1000003.0))),
    };

    let result = interp.eval(&mod_expr);
    let expected = ((12345i64.wrapping_mul(1664525) + 1013904223) % 1000003).rem_euclid(1000003);
    assert_eq!(result, LatticeValue::Constant(Literal::Int(expected)));
}

#[test]
fn test_sccp_prune_opaque_predicate_true() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // r0 = 10
    // r1 = 20
    // if (r0 + r1 == 30) then Block 1 (Safe) else Block 2 (Trap)
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (1, 1),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Binary {
                    op: BinOp::Add,
                    left: Box::leak(Box::new(IrExpr::Register(0))),
                    right: Box::leak(Box::new(IrExpr::Register(1))),
                })),
                right: Box::leak(Box::new(IrExpr::Number(30.0))),
            },
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(10.0),
    });
    b0.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(20.0),
    });

    // Block 1 (Safe):
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(0)]);
    let mut b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Return(ret_vals));
    b1.instructions.push(IrInst::Assign {
        dest: 2,
        src: IrExpr::Number(100.0),
    });

    // Block 2 (Trap / LPH_CRASH):
    let b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Crash);

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.recompute_successors();

    assert_eq!(cfg.blocks.len(), 3);
    assert!(cfg.blocks.contains_key(&BlockId(2)), "Trap block exists before SCCP");

    let mut sccp = SccpEngine::new(&mut cfg);
    sccp.run();

    // After SCCP, Block 0's branch should be converted to Goto(Block 1)
    let b0_after = &sccp.cfg.blocks[&BlockId::ENTRY];
    assert_eq!(
        b0_after.terminator,
        Terminator::Goto(BlockId(1)),
        "Branch must be rewritten to Goto"
    );

    // Trap block 2 must be pruned as dead code
    assert!(
        !sccp.cfg.blocks.contains_key(&BlockId(2)),
        "Dead trap block must be pruned by SCCP"
    );
}

#[test]
fn test_sccp_prune_opaque_predicate_false() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0:
    // r0 = 5
    // if (r0 > 100) then Block 1 (Trap) else Block 2 (Safe)
    let mut b0 = BasicBlock::new(
        BlockId(0),
        (1, 1),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Gt,
                left: Box::leak(Box::new(IrExpr::Register(0))),
                right: Box::leak(Box::new(IrExpr::Number(100.0))),
            },
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(5.0),
    });

    // Block 1 (Trap)
    let b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Crash);

    // Block 2 (Safe)
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.recompute_successors();

    let mut sccp = SccpEngine::new(&mut cfg);
    sccp.run();

    // Block 0 should jump to Block 2
    let b0_after = &sccp.cfg.blocks[&BlockId::ENTRY];
    assert_eq!(b0_after.terminator, Terminator::Goto(BlockId(2)));
    assert!(!sccp.cfg.blocks.contains_key(&BlockId(1)));
}
