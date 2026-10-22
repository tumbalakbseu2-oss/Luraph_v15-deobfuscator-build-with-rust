//! CFG construction, leader identification, and graph normalization.
//!
//! Partitions instruction streams into basic blocks, wires typed control-flow
//! edges, prunes unreachable blocks, and collapses straight-line chains.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use bumpalo::Bump;

use crate::bytecode::{ConstantValue, Instruction, OperandKind, ProtoBytecode};
use crate::cfg::ir::{
    BasicBlock, BinOp, BlockId, ControlFlowGraph, IrExpr, IrInst, Terminator, UnOp,
};

/// Two-pass leader analyzer for bytecode instruction streams.
pub struct LeaderAnalyzer;

impl LeaderAnalyzer {
    /// Computes the sorted list of 0-based instruction slice indices that are leaders.
    pub fn find_leaders(instructions: &[Instruction]) -> Vec<usize> {
        let n = instructions.len();
        if n == 0 {
            return Vec::new();
        }

        let mut is_leader = vec![false; n];
        // Rule 1: Instruction 0 (entry) is always a leader
        is_leader[0] = true;

        for (i, inst) in instructions.iter().enumerate() {
            // Rule 2 & 3: Unconditional Jump (Opcode 1 / JMP)
            if Self::is_unconditional_jump(inst.opcode) {
                if let Some(target_idx) = Self::resolve_jump_target(inst, i, n) {
                    is_leader[target_idx] = true; // Jump target is leader
                }
                if i + 1 < n {
                    is_leader[i + 1] = true; // Successor after termination is leader
                }
            }
            // Conditional branches (EQ = 12, LT, LE, NE, TEST, etc.)
            else if Self::is_conditional_branch(inst.opcode) {
                if let Some(target_idx) = Self::resolve_jump_target(inst, i, n) {
                    is_leader[target_idx] = true; // Branch target is leader
                }
                if i + 1 < n {
                    is_leader[i + 1] = true; // Fallthrough is leader
                }
            }
            // Return (Opcode 17)
            else if Self::is_return(inst.opcode) {
                if i + 1 < n {
                    is_leader[i + 1] = true; // Instruction after return is leader
                }
            }
        }

        is_leader
            .into_iter()
            .enumerate()
            .filter_map(|(idx, is_lead)| if is_lead { Some(idx) } else { None })
            .collect()
    }

    /// Resolves jump target to a 0-based instruction slice index.
    pub fn resolve_jump_target(
        inst: &Instruction,
        _current_idx: usize,
        total_len: usize,
    ) -> Option<usize> {
        let (kind_b, val_b) = OperandKind::from_raw(inst.raw_b());
        let (kind_c, val_c) = OperandKind::from_raw(inst.raw_c());

        let target_pc = match (kind_b, kind_c) {
            (OperandKind::RelativeJumpForward, _) => inst.pc as i64 + val_b as i64,
            (OperandKind::RelativeJumpBack, _) => inst.pc as i64 - val_b as i64,
            (_, OperandKind::RelativeJumpForward) => inst.pc as i64 + val_c as i64,
            (_, OperandKind::RelativeJumpBack) => inst.pc as i64 - val_c as i64,
            _ => {
                // Direct relative offset fallback
                if inst.b != 0 {
                    inst.pc as i64 + inst.b as i64
                } else if inst.c != 0 {
                    inst.pc as i64 + inst.c as i64
                } else {
                    return None;
                }
            }
        };

        // Bytecode PC is 1-based; convert to 0-based index
        if target_pc >= 1 && (target_pc as usize) <= total_len {
            Some((target_pc - 1) as usize)
        } else if target_pc >= 0 && (target_pc as usize) < total_len {
            Some(target_pc as usize)
        } else {
            None
        }
    }

    #[inline]
    pub fn is_unconditional_jump(opcode: u8) -> bool {
        opcode == 1
    }

    #[inline]
    pub fn is_conditional_branch(opcode: u8) -> bool {
        // Opcode 12 (EQ) and comparison family
        matches!(opcode, 12 | 22 | 23 | 24 | 25 | 26 | 57 | 100)
    }

    #[inline]
    pub fn is_return(opcode: u8) -> bool {
        opcode == 17
    }
}

/// Builds an initial Control Flow Graph from a prototype's instruction stream.
pub fn build_cfg<'arena>(
    proto: &ProtoBytecode,
    bump: &'arena Bump,
) -> ControlFlowGraph<'arena> {
    let instructions = &proto.instructions;
    if instructions.is_empty() {
        let mut cfg = ControlFlowGraph::new(proto.id, BlockId::ENTRY);
        let ret_vals: &'arena [IrExpr<'arena>] = bump.alloc_slice_clone(&[]);
        let entry_block = BasicBlock::new(BlockId::ENTRY, (0, 0), Terminator::Return(ret_vals));
        cfg.add_block(entry_block);
        return cfg;
    }

    let leaders = LeaderAnalyzer::find_leaders(instructions);
    let mut cfg = ControlFlowGraph::new(proto.id, BlockId::ENTRY);
    let mut pc_to_block: HashMap<usize, BlockId> = HashMap::with_capacity(leaders.len());

    // 1. Allocate block IDs for each leader
    for (block_idx, &start_idx) in leaders.iter().enumerate() {
        let bid = BlockId::new(block_idx as u32);
        pc_to_block.insert(start_idx, bid);
    }

    // 2. Populate instructions and assign terminators
    for (block_idx, &start_idx) in leaders.iter().enumerate() {
        let bid = BlockId::new(block_idx as u32);
        let end_idx = if block_idx + 1 < leaders.len() {
            leaders[block_idx + 1] - 1
        } else {
            instructions.len() - 1
        };

        let last_inst = &instructions[end_idx];
        let mut block = BasicBlock::new(
            bid,
            (instructions[start_idx].pc, last_inst.pc),
            Terminator::Unreachable,
        );

        let ends_in_terminator = LeaderAnalyzer::is_unconditional_jump(last_inst.opcode)
            || LeaderAnalyzer::is_conditional_branch(last_inst.opcode)
            || LeaderAnalyzer::is_return(last_inst.opcode);

        let linear_end = if ends_in_terminator { end_idx } else { end_idx + 1 };

        // Translate linear instructions
        for inst in &instructions[start_idx..linear_end] {
            if let Some(ir_inst) = translate_linear_inst(inst, &proto.constants, &proto.child_protos, bump) {
                block.instructions.push(ir_inst);
            }
        }

        // Assign terminator
        if LeaderAnalyzer::is_unconditional_jump(last_inst.opcode) {
            if let Some(target_idx) =
                LeaderAnalyzer::resolve_jump_target(last_inst, end_idx, instructions.len())
            {
                if let Some(&target_bid) = pc_to_block.get(&target_idx) {
                    block.terminator = Terminator::Goto(target_bid);
                }
            }
        } else if LeaderAnalyzer::is_conditional_branch(last_inst.opcode) {
            let true_bid = LeaderAnalyzer::resolve_jump_target(last_inst, end_idx, instructions.len())
                .and_then(|t| pc_to_block.get(&t).copied())
                .unwrap_or(BlockId::INVALID);

            let false_bid = pc_to_block
                .get(&(end_idx + 1))
                .copied()
                .unwrap_or(BlockId::INVALID);

            let cond = translate_branch_condition(last_inst, bump);
            block.terminator = Terminator::Branch {
                cond,
                true_target: true_bid,
                false_target: false_bid,
            };
        } else if LeaderAnalyzer::is_return(last_inst.opcode) {
            let ret_vals = translate_return_values(last_inst, bump);
            block.terminator = Terminator::Return(ret_vals);
        } else if block_idx + 1 < leaders.len() {
            // Natural fallthrough into adjacent basic block
            let next_bid = BlockId::new((block_idx + 1) as u32);
            block.terminator = Terminator::Goto(next_bid);
        } else {
            // Reached end of prototype without explicit return
            let ret_vals: &'arena [IrExpr<'arena>] = bump.alloc_slice_clone(&[]);
            block.terminator = Terminator::Return(ret_vals);
        }

        cfg.add_block(block);
    }

    cfg.recompute_successors();
    cfg
}

/// Translates a single bytecode instruction into a linear IR statement.
fn translate_linear_inst<'arena>(
    inst: &Instruction,
    constants: &[ConstantValue],
    child_protos: &[usize],
    bump: &'arena Bump,
) -> Option<IrInst<'arena>> {
    match inst.opcode {
        // MOVE: R(A) = R(B)
        3 => {
            let b_reg = inst.b as u16;
            Some(IrInst::Assign {
                dest: inst.a,
                src: IrExpr::Register(b_reg),
            })
        }
        // LOADK: R(A) = K(B) or closure reference
        5 => {
            let (kind_c, val_c) = OperandKind::from_raw(inst.raw_c());
            let (kind_b, val_b) = OperandKind::from_raw(inst.raw_b());
            let expr = if kind_c == OperandKind::ChildProtoIndex && val_c > 0 && (val_c as usize) <= 100 {
                let upvalues: &'arena [IrExpr<'arena>] = bump.alloc_slice_clone(&[]);
                IrExpr::Closure {
                    proto_id: val_c as usize,
                    upvalues,
                }
            } else if kind_b == OperandKind::ChildProtoIndex && val_b > 0 && (val_b as usize) <= 100 {
                let upvalues: &'arena [IrExpr<'arena>] = bump.alloc_slice_clone(&[]);
                IrExpr::Closure {
                    proto_id: val_b as usize,
                    upvalues,
                }
            } else {
                let k_idx = inst.b as usize;
                if let Some(c) = constants.get(k_idx) {
                    constant_to_ir_expr(c, bump)
                } else {
                    IrExpr::Constant(k_idx)
                }
            };
            Some(IrInst::Assign {
                dest: inst.a,
                src: expr,
            })
        }
        // LOADNIL: R(A) = nil
        18 => Some(IrInst::Assign {
            dest: inst.a,
            src: IrExpr::Nil,
        }),
        // GETGLOBAL: R(A) = _G[K(B)]
        8 => {
            let k_idx = inst.b as usize;
            let name = constants
                .get(k_idx)
                .and_then(|c| c.as_str())
                .unwrap_or("_G_var");
            let arena_name = bump.alloc_str(name);
            Some(IrInst::Assign {
                dest: inst.a,
                src: IrExpr::Global(arena_name),
            })
        }
        // SETGLOBAL: _G[K(B)] = R(A)
        9 => {
            let k_idx = inst.b as usize;
            let name = constants
                .get(k_idx)
                .and_then(|c| c.as_str())
                .unwrap_or("_G_var");
            let arena_name = bump.alloc_str(name);
            Some(IrInst::SetGlobal {
                name: arena_name,
                src: IrExpr::Register(inst.a),
            })
        }
        // GETTABLE: R(A) = R(B)[R(C)]
        6 => {
            let tbl = bump.alloc(IrExpr::Register(inst.b as u16));
            let key = bump.alloc(resolve_rk(inst.c, constants, child_protos, bump));
            Some(IrInst::Assign {
                dest: inst.a,
                src: IrExpr::TableGet { table: tbl, key },
            })
        }
        // SETTABLE: R(A)[R(B)] = R(C)
        7 => {
            let tbl = IrExpr::Register(inst.a);
            let key = resolve_rk(inst.b, constants, child_protos, bump);
            let val = resolve_rk(inst.c, constants, child_protos, bump);
            Some(IrInst::TableSet {
                table: tbl,
                key,
                value: val,
            })
        }
        // NEWTABLE: R(A) = {}
        4 => Some(IrInst::Assign {
            dest: inst.a,
            src: IrExpr::NewTable {
                array_size: inst.b.max(0) as u32,
                hash_size: inst.c.max(0) as u32,
            },
        }),
        // ADD: R(A) = R(B) + R(C)
        10 => translate_binary_op(inst, BinOp::Add, constants, child_protos, bump),
        // MOD: R(A) = R(B) % R(C)
        11 => translate_binary_op(inst, BinOp::Mod, constants, child_protos, bump),
        // CONCAT: R(A) = R(B) .. R(C)
        13 => translate_binary_op(inst, BinOp::Concat, constants, child_protos, bump),
        // LEN: R(A) = #R(B)
        14 => {
            let expr = bump.alloc(IrExpr::Register(inst.b as u16));
            Some(IrInst::Assign {
                dest: inst.a,
                src: IrExpr::Unary {
                    op: UnOp::Len,
                    expr,
                },
            })
        }
        // NOT: R(A) = not R(B)
        15 => {
            let expr = bump.alloc(IrExpr::Register(inst.b as u16));
            Some(IrInst::Assign {
                dest: inst.a,
                src: IrExpr::Unary {
                    op: UnOp::Not,
                    expr,
                },
            })
        }
        // GETUPVAL: R(A) = UP(B)
        20 => Some(IrInst::Assign {
            dest: inst.a,
            src: IrExpr::Upvalue(inst.b as u16),
        }),
        // SETUPVAL: UP(B) = R(A)
        21 => Some(IrInst::SetUpvalue {
            upvalue_idx: inst.b as u16,
            src: IrExpr::Register(inst.a),
        }),
        // CLOSURE: R(A) = closure(K(B))
        16 => {
            let proto_id = if let Some(&cid) = child_protos.get(inst.b as usize) {
                cid
            } else {
                inst.b as usize
            };
            let upvalues: &'arena [IrExpr<'arena>] = bump.alloc_slice_clone(&[]);
            Some(IrInst::Assign {
                dest: inst.a,
                src: IrExpr::Closure {
                    proto_id,
                    upvalues,
                },
            })
        }
        // CALL: R(A)(R(A+1)...)
        2 => {
            let callee = bump.alloc(IrExpr::Register(inst.a));
            let arg_count = if inst.b > 0 { ((inst.b - 1) as usize).min(16) } else { 0 };
            let mut args = Vec::with_capacity(arg_count);
            for offset in 1..=arg_count {
                args.push(IrExpr::Register(inst.a.wrapping_add(offset as u16)));
            }
            let arena_args: &'arena [IrExpr<'arena>] = bump.alloc_slice_clone(&args);
            Some(IrInst::CallStmt {
                callee: callee.clone(),
                args: arena_args,
                ret_dest: if inst.c != 1 { Some(inst.a) } else { None },
                ret_count: inst.c - 1,
            })
        }
        // VARARG: R(A)... = ...
        19 => Some(IrInst::Assign {
            dest: inst.a,
            src: IrExpr::Vararg,
        }),
        _ => None,
    }
}

fn translate_binary_op<'arena>(
    inst: &Instruction,
    op: BinOp,
    constants: &[ConstantValue],
    child_protos: &[usize],
    bump: &'arena Bump,
) -> Option<IrInst<'arena>> {
    let left = bump.alloc(resolve_rk(inst.b, constants, child_protos, bump));
    let right = bump.alloc(resolve_rk(inst.c, constants, child_protos, bump));
    Some(IrInst::Assign {
        dest: inst.a,
        src: IrExpr::Binary { op, left, right },
    })
}

fn resolve_rk<'arena>(
    val: i32,
    constants: &[ConstantValue],
    child_protos: &[usize],
    bump: &'arena Bump,
) -> IrExpr<'arena> {
    let (kind, operand_val) = OperandKind::from_raw(val);
    match kind {
        OperandKind::ConstantIndex => {
            let idx = operand_val.max(0) as usize;
            if let Some(c) = constants.get(idx) {
                constant_to_ir_expr(c, bump)
            } else {
                IrExpr::Constant(idx)
            }
        }
        OperandKind::ChildProtoIndex => {
            let pid = operand_val.max(0) as usize;
            let target_pid = if child_protos.contains(&pid) {
                pid
            } else if let Some(&cid) = child_protos.get(pid) {
                cid
            } else if pid > 0 && pid <= 60 {
                pid
            } else {
                0
            };
            if target_pid > 0 {
                let upvalues: &'arena [IrExpr<'arena>] = bump.alloc_slice_clone(&[]);
                IrExpr::Closure {
                    proto_id: target_pid,
                    upvalues,
                }
            } else {
                IrExpr::Register(operand_val.max(0) as u16)
            }
        }
        _ => IrExpr::Register(operand_val.max(0) as u16),
    }
}

fn constant_to_ir_expr<'arena>(c: &ConstantValue, bump: &'arena Bump) -> IrExpr<'arena> {
    match c {
        ConstantValue::Nil => IrExpr::Nil,
        ConstantValue::Boolean(b) => IrExpr::Boolean(*b),
        ConstantValue::Number(n) => IrExpr::Number(*n),
        ConstantValue::Float32(f) => IrExpr::Number(*f as f64),
        ConstantValue::Int64(i) => IrExpr::Number(*i as f64),
        ConstantValue::String(s) => {
            let arena_str = bump.alloc_str(s.as_ref());
            IrExpr::String(arena_str)
        }
        _ => IrExpr::Nil,
    }
}

fn translate_branch_condition<'arena>(inst: &Instruction, bump: &'arena Bump) -> IrExpr<'arena> {
    let left = bump.alloc(IrExpr::Register(inst.a));
    let right = bump.alloc(IrExpr::Register(inst.b.max(0) as u16));
    let op = match inst.opcode {
        12 => BinOp::Eq,
        22 => BinOp::Ne,
        23 => BinOp::Lt,
        24 => BinOp::Le,
        25 => BinOp::Gt,
        26 => BinOp::Ge,
        _ => BinOp::Eq,
    };
    IrExpr::Binary { op, left, right }
}

fn translate_return_values<'arena>(
    inst: &Instruction,
    bump: &'arena Bump,
) -> &'arena [IrExpr<'arena>] {
    let count = if inst.b > 0 { ((inst.b - 1) as usize).min(8) } else { 0 };
    let mut vals = Vec::with_capacity(count);
    for i in 0..count {
        vals.push(IrExpr::Register(inst.a + i as u16));
    }
    bump.alloc_slice_clone(&vals)
}

/// Prunes all blocks that are unreachable from the CFG entry block.
pub fn prune_unreachable<'arena>(cfg: &mut ControlFlowGraph<'arena>) {
    if !cfg.blocks.contains_key(&cfg.entry) {
        if let Some(&first_bid) = cfg.blocks.keys().next() {
            cfg.entry = first_bid;
        } else {
            cfg.blocks.clear();
            return;
        }
    }

    let mut reachable = HashSet::with_capacity(cfg.blocks.len());
    let mut queue = VecDeque::new();
    queue.push_back(cfg.entry);

    while let Some(bid) = queue.pop_front() {
        if reachable.insert(bid) {
            if let Some(block) = cfg.blocks.get(&bid) {
                for &succ in &block.successors {
                    if !reachable.contains(&succ) {
                        queue.push_back(succ);
                    }
                }
            }
        }
    }

    cfg.blocks.retain(|bid, _| reachable.contains(bid));
    cfg.recompute_predecessors();
}

/// Trampoline contraction, branch deduplication, and straight-line chain collapsing.
pub fn thread_empty<'arena>(cfg: &mut ControlFlowGraph<'arena>) {
    // 1. Resolve trampoline chains (empty Goto blocks)
    fn resolve_target(
        mut bid: BlockId,
        blocks: &indexmap::IndexMap<BlockId, BasicBlock>,
    ) -> BlockId {
        let mut seen = HashSet::new();
        while let Some(b) = blocks.get(&bid) {
            if b.is_empty_trampoline() {
                if let Terminator::Goto(next_bid) = b.terminator {
                    if next_bid != bid && seen.insert(bid) {
                        bid = next_bid;
                        continue;
                    }
                }
            }
            break;
        }
        bid
    }

    // Precompute target resolution map for all blocks to avoid borrow conflict
    let mut resolved_map: HashMap<BlockId, BlockId> = HashMap::with_capacity(cfg.blocks.len());
    for &bid in cfg.blocks.keys() {
        resolved_map.insert(bid, resolve_target(bid, &cfg.blocks));
    }

    cfg.entry = resolved_map.get(&cfg.entry).copied().unwrap_or(cfg.entry);

    // Redirect terminators and successors
    for block in cfg.blocks.values_mut() {
        match &mut block.terminator {
            Terminator::Goto(target) => {
                if let Some(&new_t) = resolved_map.get(target) {
                    *target = new_t;
                }
            }
            Terminator::Branch {
                true_target,
                false_target,
                ..
            } => {
                if let Some(&new_t) = resolved_map.get(true_target) {
                    *true_target = new_t;
                }
                if let Some(&new_f) = resolved_map.get(false_target) {
                    *false_target = new_f;
                }
            }
            _ => {}
        }
    }

    // Convert branches with identical targets to Goto
    for block in cfg.blocks.values_mut() {
        if let Terminator::Branch {
            true_target,
            false_target,
            ..
        } = block.terminator
        {
            if true_target == false_target {
                block.terminator = Terminator::Goto(true_target);
            }
        }
    }

    cfg.recompute_successors();
    prune_unreachable(cfg);

    // 2. Straight-line chain collapsing loop
    let mut changed = true;
    while changed {
        changed = false;
        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        for a_id in block_ids {
            if !cfg.blocks.contains_key(&a_id) {
                continue;
            }

            let b_id = match cfg.blocks[&a_id].terminator {
                Terminator::Goto(target) => target,
                _ => continue,
            };

            if b_id == a_id || b_id == cfg.entry {
                continue;
            }

            // Check if B has exactly one predecessor (A)
            let b_single_pred = cfg
                .blocks
                .get(&b_id)
                .map(|b| b.predecessors.len() == 1 && b.predecessors[0] == a_id)
                .unwrap_or(false);

            if b_single_pred {
                // Remove B and merge its instructions and terminator into A
                if let Some(b_block) = cfg.blocks.shift_remove(&b_id) {
                    let a_block = cfg.blocks.get_mut(&a_id).unwrap();
                    a_block.instructions.extend(b_block.instructions);
                    a_block.terminator = b_block.terminator;
                    a_block.pc_range.1 = b_block.pc_range.1;
                    changed = true;
                }
            }
        }

        if changed {
            cfg.recompute_successors();
            prune_unreachable(cfg);
        }
    }
}

/// Block deduplication via partition refinement.
/// Collapses identical overlay-duplicated paths into canonical representatives.
pub fn merge_equivalent<'arena>(cfg: &mut ControlFlowGraph<'arena>) -> bool {
    let mut merged_any = false;

    loop {
        if cfg.blocks.len() <= 1 {
            break;
        }

        // Step 1: Compute block signature texts (preserving deterministic BlockId order)
        let mut signatures: BTreeMap<BlockId, String> = BTreeMap::new();
        for (&bid, b) in &cfg.blocks {
            signatures.insert(bid, compute_block_signature(b));
        }

        // Step 2: Partition refinement using ordered maps
        let mut block_to_class: BTreeMap<BlockId, usize> = BTreeMap::new();
        let mut sig_to_class: BTreeMap<String, usize> = BTreeMap::new();

        for (&bid, sig) in &signatures {
            let next_id = sig_to_class.len();
            let cls = *sig_to_class.entry(sig.clone()).or_insert(next_id);
            block_to_class.insert(bid, cls);
        }

        let mut num_classes = sig_to_class.len();
        loop {
            let mut refined_classes: BTreeMap<(usize, Vec<usize>), usize> = BTreeMap::new();
            let mut new_block_to_class: BTreeMap<BlockId, usize> = BTreeMap::new();

            for (&bid, b) in &cfg.blocks {
                let current_class = block_to_class[&bid];
                let succ_classes: Vec<usize> = b
                    .successors
                    .iter()
                    .filter_map(|succ| block_to_class.get(succ).copied())
                    .collect();

                let key = (current_class, succ_classes);
                let next_cls = refined_classes.len();
                let cls = *refined_classes.entry(key).or_insert(next_cls);
                new_block_to_class.insert(bid, cls);
            }

            if refined_classes.len() == num_classes {
                break; // Fixpoint reached
            }
            num_classes = refined_classes.len();
            block_to_class = new_block_to_class;
        }

        // Step 3: Group by class and merge duplicates in deterministic class order
        let mut class_to_blocks: BTreeMap<usize, Vec<BlockId>> = BTreeMap::new();
        for (&bid, &cls) in &block_to_class {
            class_to_blocks.entry(cls).or_default().push(bid);
        }

        let mut replacements: BTreeMap<BlockId, BlockId> = BTreeMap::new();
        for blocks in class_to_blocks.values_mut() {
            if blocks.len() > 1 {
                blocks.sort_by_key(|b| b.0);
                // Canonical representative: entry if present, else minimum BlockId
                let canonical = if blocks.contains(&cfg.entry) {
                    cfg.entry
                } else {
                    blocks[0]
                };

                for &b in blocks.iter() {
                    if b != canonical {
                        replacements.insert(b, canonical);
                    }
                }
            }
        }

        if replacements.is_empty() {
            break;
        }

        merged_any = true;

        // Redirect all references
        for block in cfg.blocks.values_mut() {
            match &mut block.terminator {
                Terminator::Goto(target) => {
                    if let Some(&rep) = replacements.get(target) {
                        *target = rep;
                    }
                }
                Terminator::Branch {
                    true_target,
                    false_target,
                    ..
                } => {
                    if let Some(&rep) = replacements.get(true_target) {
                        *true_target = rep;
                    }
                    if let Some(&rep) = replacements.get(false_target) {
                        *false_target = rep;
                    }
                }
                _ => {}
            }
        }

        // Remove redundant blocks in deterministic sorted BlockId order
        let mut dead_ids: Vec<BlockId> = replacements.keys().copied().collect();
        dead_ids.sort_by_key(|b| b.0);
        for dead_id in dead_ids {
            cfg.blocks.shift_remove(&dead_id);
        }

        cfg.recompute_successors();
        prune_unreachable(cfg);
        thread_empty(cfg);
    }

    merged_any
}

/// Computes a structural signature for a basic block.
fn compute_block_signature(block: &BasicBlock) -> String {
    let mut sig = String::new();
    for inst in &block.instructions {
        match inst {
            IrInst::Assign { dest, .. } => sig.push_str(&format!("A{dest};")),
            IrInst::TableSet { .. } => sig.push_str("TSet;"),
            IrInst::SetUpvalue { upvalue_idx, .. } => sig.push_str(&format!("UP{upvalue_idx};")),
            IrInst::SetGlobal { name, .. } => sig.push_str(&format!("G{name};")),
            IrInst::CallStmt { ret_count, .. } => sig.push_str(&format!("Call{ret_count};")),
            IrInst::SetList { start_idx, .. } => sig.push_str(&format!("SetList{start_idx};")),
            IrInst::CloseUpvalue(reg) => sig.push_str(&format!("Close{reg};")),
            IrInst::Comment(_) | IrInst::Nop => {}
        }
    }
    match &block.terminator {
        Terminator::Goto(_) => sig.push_str("Goto"),
        Terminator::Branch { .. } => sig.push_str("Branch"),
        Terminator::Return(_) => sig.push_str("Ret"),
        Terminator::ForLoop { .. } => sig.push_str("For"),
        Terminator::ForInLoop { .. } => sig.push_str("ForIn"),
        Terminator::Crash => sig.push_str("Crash"),
        Terminator::Unreachable => sig.push_str("Unreach"),
    }
    sig
}
