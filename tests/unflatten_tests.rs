//! Unit tests for Control-Flow Unflattening and Central VM Dispatcher Elimination.

use bumpalo::Bump;
use luraph_deobf_rs::cfg::{
    BasicBlock, BinOp, BlockId, ControlFlowGraph, IrExpr, IrInst, Terminator, Unflattener,
};

#[test]
fn test_dispatcher_detection_and_unflattening() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0 (Entry):
    // state (r99) = 1
    // goto Dispatcher (Block 10)
    let mut b0 = BasicBlock::new(BlockId(0), (1, 1), Terminator::Goto(BlockId(10)));
    b0.instructions.push(IrInst::Assign {
        dest: 99,
        src: IrExpr::Number(1.0),
    });

    // Block 10 (Dispatcher Header):
    // if state == 1 then goto Handler 1 (Block 1) else goto Block 11
    let b10 = BasicBlock::new(
        BlockId(10),
        (2, 2),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(99))),
                right: Box::leak(Box::new(IrExpr::Number(1.0))),
            },
            true_target: BlockId(1),
            false_target: BlockId(11),
        },
    );

    // Block 11 (Dispatcher Tree Branch):
    // if state == 2 then goto Handler 2 (Block 2) else goto Crash (Block 99)
    let b11 = BasicBlock::new(
        BlockId(11),
        (3, 3),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(99))),
                right: Box::leak(Box::new(IrExpr::Number(2.0))),
            },
            true_target: BlockId(2),
            false_target: BlockId(99),
        },
    );

    // Block 1 (Handler 1):
    // r0 = 42
    // state (r99) = 2
    // goto Dispatcher (Block 10)
    let mut b1 = BasicBlock::new(BlockId(1), (4, 4), Terminator::Goto(BlockId(10)));
    b1.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(42.0),
    });
    b1.instructions.push(IrInst::Assign {
        dest: 99,
        src: IrExpr::Number(2.0),
    });

    // Block 2 (Handler 2):
    // r1 = r0 + 1
    // return r1
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(1)]);
    let mut b2 = BasicBlock::new(BlockId(2), (5, 5), Terminator::Return(ret_vals));
    b2.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Binary {
            op: BinOp::Add,
            left: Box::leak(Box::new(IrExpr::Register(0))),
            right: Box::leak(Box::new(IrExpr::Number(1.0))),
        },
    });

    // Block 99 (Crash Stub):
    let b99 = BasicBlock::new(BlockId(99), (6, 6), Terminator::Crash);

    cfg.add_block(b0);
    cfg.add_block(b10);
    cfg.add_block(b11);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b99);
    cfg.recompute_successors();

    assert_eq!(cfg.blocks.len(), 6);

    // 1. Detect Dispatcher
    let disp_opt = Unflattener::detect_dispatcher(&cfg);
    assert!(disp_opt.is_some(), "Must detect central VM dispatcher");
    let disp = disp_opt.unwrap();
    assert_eq!(disp.header, BlockId(10));
    assert_eq!(disp.state_reg, Some(99));
    assert_eq!(disp.state_map.get(&1), Some(&BlockId(1)));
    assert_eq!(disp.state_map.get(&2), Some(&BlockId(2)));

    // 2. Perform Unflattening
    Unflattener::unflatten(&mut cfg, &disp);

    // 3. Verify results:
    // - Dispatcher blocks 10 and 11 must be pruned
    assert!(!cfg.blocks.contains_key(&BlockId(10)), "Dispatcher header must be pruned");
    assert!(!cfg.blocks.contains_key(&BlockId(11)), "Dispatcher tree node must be pruned");
    assert!(!cfg.blocks.contains_key(&BlockId(99)), "Crash stub must be pruned");

    // - The sequence has unflattened: dead state variable assignments (dest 99) are eliminated
    for block in cfg.blocks.values() {
        assert!(
            !block.instructions.iter().any(|inst| matches!(inst, IrInst::Assign { dest: 99, .. })),
            "Dead state variable assignment must be removed"
        );
    }

    // - Entry block now reaches the terminal return cleanly
    let entry = &cfg.blocks[&cfg.entry];
    assert!(
        matches!(entry.terminator, Terminator::Return(_) | Terminator::Goto(_)),
        "Entry must reach unflattened handler"
    );
}

#[test]
fn test_unflatten_conditional_branching() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Entry (0): state = 1, goto 10
    let mut b0 = BasicBlock::new(BlockId(0), (1, 1), Terminator::Goto(BlockId(10)));
    b0.instructions.push(IrInst::Assign {
        dest: 99,
        src: IrExpr::Number(1.0),
    });

    // Dispatcher (10):
    // if state == 1 then goto 1 else goto 11
    let b10 = BasicBlock::new(
        BlockId(10),
        (2, 2),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(99))),
                right: Box::leak(Box::new(IrExpr::Number(1.0))),
            },
            true_target: BlockId(1),
            false_target: BlockId(11),
        },
    );

    // Dispatcher branch (11):
    // if state == 2 then goto 2 else goto 3
    let b11 = BasicBlock::new(
        BlockId(11),
        (3, 3),
        Terminator::Branch {
            cond: IrExpr::Binary {
                op: BinOp::Eq,
                left: Box::leak(Box::new(IrExpr::Register(99))),
                right: Box::leak(Box::new(IrExpr::Number(2.0))),
            },
            true_target: BlockId(2),
            false_target: BlockId(3),
        },
    );

    // Handler 1: branches to either state 2 or state 3
    let b1 = BasicBlock::new(
        BlockId(1),
        (4, 4),
        Terminator::Branch {
            cond: IrExpr::Register(5),
            true_target: BlockId(20), // sets state=2 and jumps to 10
            false_target: BlockId(21), // sets state=3 and jumps to 10
        },
    );

    let mut b20 = BasicBlock::new(BlockId(20), (5, 5), Terminator::Goto(BlockId(10)));
    b20.instructions.push(IrInst::Assign {
        dest: 99,
        src: IrExpr::Number(2.0),
    });

    let mut b21 = BasicBlock::new(BlockId(21), (6, 6), Terminator::Goto(BlockId(10)));
    b21.instructions.push(IrInst::Assign {
        dest: 99,
        src: IrExpr::Number(3.0),
    });

    // Handler 2 (returns 10)
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Number(10.0)]);
    let b2 = BasicBlock::new(BlockId(2), (7, 7), Terminator::Return(ret_vals));

    // Handler 3 (returns 20)
    let ret_vals3: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Number(20.0)]);
    let b3 = BasicBlock::new(BlockId(3), (8, 8), Terminator::Return(ret_vals3));

    cfg.add_block(b0);
    cfg.add_block(b10);
    cfg.add_block(b11);
    cfg.add_block(b1);
    cfg.add_block(b20);
    cfg.add_block(b21);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    let disp = Unflattener::detect_dispatcher(&cfg).expect("Dispatcher must be detected");
    Unflattener::unflatten(&mut cfg, &disp);

    // Dispatcher tree blocks 10 and 11 must be eliminated
    assert!(!cfg.blocks.contains_key(&BlockId(10)));
    assert!(!cfg.blocks.contains_key(&BlockId(11)));
}
