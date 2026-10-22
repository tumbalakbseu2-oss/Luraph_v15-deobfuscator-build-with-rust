//! Comprehensive unit tests for CFG construction, leader detection,
//! dominators, natural loops, and region structuring.

use bumpalo::Bump;
use luraph_deobf_rs::bytecode::{Instruction, ProtoBytecode};
use luraph_deobf_rs::cfg::{
    build_cfg, merge_equivalent, prune_unreachable, thread_empty, BasicBlock, BinOp, BlockId,
    ControlFlowGraph, DominatorTree, IrExpr, IrInst, LeaderAnalyzer, LoopExtractor, LoopKind,
    PostDominatorTree, RegionStructurer, StructuredRegion, Terminator,
};

#[test]
fn test_leader_identification_basic() {
    // Instruction stream:
    // 0: MOVE (entry) -> leader
    // 1: JMP to PC 4 (index 3)
    // 2: ADD (successor of jump) -> leader
    // 3: SUB (jump target) -> leader
    // 4: RETURN -> successor if any is leader
    let insts = vec![
        Instruction::new(1, 3, 0, 1, 0, [0, 0, 0, 0]),
        // Opcode 1 (JMP), raw_b = 2 * 8 + 3 (Mode 3: RelativeJumpForward, target = 2 + 2 = 4 -> index 3)
        Instruction::new(2, 1, 0, 2, 0, [0, 0, 0, 2 * 8 + 3]),
        Instruction::new(3, 10, 0, 1, 2, [0, 0, 0, 0]),
        Instruction::new(4, 10, 0, 1, 2, [0, 0, 0, 0]),
        Instruction::new(5, 17, 0, 0, 0, [0, 0, 0, 0]),
    ];

    let leaders = LeaderAnalyzer::find_leaders(&insts);
    assert!(leaders.contains(&0), "Entry must be a leader");
    assert!(leaders.contains(&2), "Instruction after JMP must be leader");
    assert!(leaders.contains(&3), "Jump target must be leader");
}

#[test]
fn test_straight_line_block_collapse() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Block 0: Goto(1)
    let mut b0 = BasicBlock::new(BlockId(0), (1, 1), Terminator::Goto(BlockId(1)));
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(10.0),
    });

    // Block 1: single predecessor (0), Goto(2)
    let mut b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Goto(BlockId(2)));
    b1.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(20.0),
    });

    // Block 2: single predecessor (1), Return
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(1)]);
    let mut b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Return(ret_vals));
    b2.instructions.push(IrInst::Assign {
        dest: 2,
        src: IrExpr::Number(30.0),
    });

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.recompute_successors();

    assert_eq!(cfg.blocks.len(), 3);

    // Run thread_empty to collapse straight-line single-predecessor blocks
    thread_empty(&mut cfg);

    // After collapsing, Block 1 and Block 2 should be folded into Block 0!
    assert_eq!(cfg.blocks.len(), 1, "Straight-line blocks must collapse into single block");
    let collapsed = &cfg.blocks[&BlockId::ENTRY];
    assert_eq!(collapsed.instructions.len(), 3);
    assert!(matches!(collapsed.terminator, Terminator::Return(_)));
}

#[test]
fn test_trampoline_threading() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Entry: Goto(1)
    let mut b0 = BasicBlock::new(BlockId(0), (1, 1), Terminator::Goto(BlockId(1)));
    b0.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(42.0),
    });

    // Block 1: Empty trampoline jumping to Block 2
    let b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Goto(BlockId(2)));

    // Block 2: Empty trampoline jumping to Block 3
    let b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Goto(BlockId(3)));

    // Block 3: Target block
    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b3 = BasicBlock::new(BlockId(3), (4, 4), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    thread_empty(&mut cfg);

    // Block 0 should now point directly to Block 3 or be collapsed
    assert!(!cfg.blocks.contains_key(&BlockId(1)));
    assert!(!cfg.blocks.contains_key(&BlockId(2)));
}

#[test]
fn test_dominator_tree_diamond() {
    // Diamond CFG:
    //      0 (Entry)
    //     / \
    //    1   2
    //     \ /
    //      3 (Join)
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let b0 = BasicBlock::new(
        BlockId(0),
        (1, 1),
        Terminator::Branch {
            cond: IrExpr::Boolean(true),
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );
    let mut b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Goto(BlockId(3)));
    b1.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(10.0),
    });

    let mut b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Goto(BlockId(3)));
    b2.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(20.0),
    });

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b3 = BasicBlock::new(BlockId(3), (4, 4), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    let dom = DominatorTree::compute(&cfg);
    assert_eq!(dom.idom.get(&BlockId(0)), Some(&BlockId(0)));
    assert_eq!(dom.idom.get(&BlockId(1)), Some(&BlockId(0)));
    assert_eq!(dom.idom.get(&BlockId(2)), Some(&BlockId(0)));
    assert_eq!(dom.idom.get(&BlockId(3)), Some(&BlockId(0)));

    assert!(dom.dominates(BlockId(0), BlockId(3)));
    assert!(!dom.dominates(BlockId(1), BlockId(3)));
    assert!(!dom.dominates(BlockId(2), BlockId(3)));

    let pdom = PostDominatorTree::compute(&cfg);
    assert_eq!(pdom.ipdom.get(&BlockId(0)), Some(&BlockId(3)));
    assert_eq!(pdom.ipdom.get(&BlockId(1)), Some(&BlockId(3)));
    assert_eq!(pdom.ipdom.get(&BlockId(2)), Some(&BlockId(3)));
}

#[test]
fn test_natural_loop_detection() {
    // Loop CFG:
    // 0 (Entry) -> 1 (Header)
    // 1 -> Branch(2, 3)
    // 2 (Body/Latch) -> Goto(1) [Back-edge]
    // 3 (Exit) -> Return
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let b0 = BasicBlock::new(BlockId(0), (1, 1), Terminator::Goto(BlockId(1)));
    let b1 = BasicBlock::new(
        BlockId(1),
        (2, 2),
        Terminator::Branch {
            cond: IrExpr::Boolean(true),
            true_target: BlockId(2),
            false_target: BlockId(3),
        },
    );
    let mut b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Goto(BlockId(1)));
    b2.instructions.push(IrInst::Assign {
        dest: 0,
        src: IrExpr::Number(1.0),
    });

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b3 = BasicBlock::new(BlockId(3), (4, 4), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    let dom = DominatorTree::compute(&cfg);
    assert!(dom.dominates(BlockId(1), BlockId(2)));

    let loops = LoopExtractor::extract_loops(&cfg, &dom);
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].header, BlockId(1));
    assert!(loops[0].latches.contains(&BlockId(2)));
    assert!(loops[0].body.contains(&BlockId(1)));
    assert!(loops[0].body.contains(&BlockId(2)));
}

#[test]
fn test_region_structuring_diamond_if() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    let b0 = BasicBlock::new(
        BlockId(0),
        (1, 1),
        Terminator::Branch {
            cond: IrExpr::Register(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );
    let mut b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Goto(BlockId(3)));
    b1.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(100.0),
    });

    let mut b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Goto(BlockId(3)));
    b2.instructions.push(IrInst::Assign {
        dest: 1,
        src: IrExpr::Number(200.0),
    });

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[IrExpr::Register(1)]);
    let b3 = BasicBlock::new(BlockId(3), (4, 4), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    let mut structurer = RegionStructurer::new(&cfg);
    let region = structurer.structure();

    // Verify structured region contains an If
    let mut found_if = false;
    fn search_if(r: &StructuredRegion, found: &mut bool) {
        match r {
            StructuredRegion::If { .. } => *found = true,
            StructuredRegion::Seq(v) => {
                for sub in v {
                    search_if(sub, found);
                }
            }
            _ => {}
        }
    }
    search_if(&region, &mut found_if);
    assert!(found_if, "Structuring must recover an If region for diamond control flow");
}

#[test]
fn test_block_deduplication_merge_equivalent() {
    let bump = Bump::new();
    let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

    // Entry branches to identical blocks 1 and 2
    let b0 = BasicBlock::new(
        BlockId(0),
        (1, 1),
        Terminator::Branch {
            cond: IrExpr::Register(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        },
    );

    // Block 1 and Block 2 have identical instructions and identical target (Block 3)
    let mut b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Goto(BlockId(3)));
    b1.instructions.push(IrInst::Assign {
        dest: 5,
        src: IrExpr::Number(999.0),
    });

    let mut b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Goto(BlockId(3)));
    b2.instructions.push(IrInst::Assign {
        dest: 5,
        src: IrExpr::Number(999.0),
    });

    let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
    let b3 = BasicBlock::new(BlockId(3), (4, 4), Terminator::Return(ret_vals));

    cfg.add_block(b0);
    cfg.add_block(b1);
    cfg.add_block(b2);
    cfg.add_block(b3);
    cfg.recompute_successors();

    assert_eq!(cfg.blocks.len(), 4);
    let merged = merge_equivalent(&mut cfg);
    assert!(merged, "Equivalent blocks 1 and 2 must be merged");
    assert!(cfg.blocks.len() < 4, "Block count must decrease after deduplication");
}
