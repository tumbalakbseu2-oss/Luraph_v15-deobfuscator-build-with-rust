//! Control-Flow Unflattening & Loop Dispatcher Elimination.
//!
//! Detects central VM dispatch loops (`while state ~= EXIT do ...`), matches handler
//! exit states to target basic blocks, rewires jumps directly to handlers bypassing
//! the dispatcher comparison tree, and contracts single-jump trampolines.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::cfg::builder::{prune_unreachable, thread_empty};
use crate::cfg::ir::{
    BinOp, BlockId, ControlFlowGraph, IrExpr, IrInst, Terminator,
};

/// Information identifying a central VM control-flow dispatcher.
#[derive(Clone, Debug)]
pub struct DispatcherInfo {
    /// Dispatcher loop header block ID.
    pub header: BlockId,
    /// Register slot holding the discrete state integer, if register-based.
    pub state_reg: Option<u16>,
    /// Mapping from state integer value to destination handler BlockId.
    pub state_map: HashMap<i64, BlockId>,
    /// Optional exit block ID when state matches exit sentinel.
    pub exit_block: Option<BlockId>,
}

pub struct Unflattener;

impl Unflattener {
    /// Detects whether the CFG contains a central control-flow flattening dispatcher.
    pub fn detect_dispatcher<'arena>(cfg: &ControlFlowGraph<'arena>) -> Option<DispatcherInfo> {
        if cfg.blocks.len() < 3 {
            return None;
        }

        // 1. Identify candidate state registers and candidate state values
        let mut reg_candidates: HashMap<u16, usize> = HashMap::new();
        let mut state_values: BTreeSet<i64> = BTreeSet::new();

        for block in cfg.blocks.values() {
            // Collect from branch conditions
            if let Terminator::Branch { cond, .. } = &block.terminator {
                if let Some((reg, val)) = Self::extract_state_comparison(cond) {
                    *reg_candidates.entry(reg).or_default() += 1;
                    state_values.insert(val);
                }
            }
            // Collect from state assignments
            for inst in &block.instructions {
                if let IrInst::Assign { dest, src } = inst {
                    if let IrExpr::Number(n) = src {
                        if n.fract() == 0.0 {
                            state_values.insert(*n as i64);
                        }
                    }
                    if let IrExpr::Register(r) = src {
                        *reg_candidates.entry(*r).or_default() += 1;
                    }
                    *reg_candidates.entry(*dest).or_default() += 1;
                }
            }
        }

        // Add standard small integers to state values
        for s in 0..=32 {
            state_values.insert(s);
        }

        // 2. Test candidate registers by order of occurrence frequency
        let mut candidates: Vec<(u16, usize)> = reg_candidates.into_iter().collect();
        candidates.sort_by_key(|&(reg, count)| (std::cmp::Reverse(count), reg));

        for (state_reg, _) in candidates {
            // Find candidate dispatcher headers: blocks with >= 2 predecessors that compare state_reg
            for (&bid, block) in &cfg.blocks {
                if block.predecessors.len() >= 2 {
                    if let Terminator::Branch { cond, .. } = &block.terminator {
                        if let Some((reg, _)) = Self::extract_state_comparison(cond) {
                            if reg == state_reg {
                                // Simulate routing for all state values through the comparison tree
                                let mut state_map = HashMap::new();
                                for &val in &state_values {
                                    if let Some(target) = Self::route_state(cfg, bid, state_reg, val) {
                                        if target != bid {
                                            state_map.insert(val, target);
                                        }
                                    }
                                }

                                if state_map.len() >= 2 {
                                    return Some(DispatcherInfo {
                                        header: bid,
                                        state_reg: Some(state_reg),
                                        state_map,
                                        exit_block: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Evaluates a branch condition given a concrete state variable value.
    fn eval_condition_with_state(cond: &IrExpr, state_reg: u16, state_val: i64) -> Option<bool> {
        match cond {
            IrExpr::Binary { op, left, right } => {
                let left_v = match &**left {
                    IrExpr::Register(r) if *r == state_reg => Some(state_val),
                    IrExpr::Number(n) => Some(*n as i64),
                    _ => None,
                };
                let right_v = match &**right {
                    IrExpr::Register(r) if *r == state_reg => Some(state_val),
                    IrExpr::Number(n) => Some(*n as i64),
                    _ => None,
                };

                match (left_v, right_v) {
                    (Some(l), Some(r)) => match op {
                        BinOp::Eq => Some(l == r),
                        BinOp::Ne => Some(l != r),
                        BinOp::Lt => Some(l < r),
                        BinOp::Le => Some(l <= r),
                        BinOp::Gt => Some(l > r),
                        BinOp::Ge => Some(l >= r),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Traces the path through the dispatcher decision tree for a given state value.
    fn route_state<'arena>(
        cfg: &ControlFlowGraph<'arena>,
        start: BlockId,
        state_reg: u16,
        state_val: i64,
    ) -> Option<BlockId> {
        let mut cur = start;
        let mut visited = HashSet::new();

        while visited.insert(cur) {
            let block = cfg.blocks.get(&cur)?;

            // If we have transitioned away from start and this block does not branch on state_reg, it is a handler!
            if cur != start {
                if let Terminator::Branch { cond, .. } = &block.terminator {
                    if Self::extract_state_comparison(cond).map(|(r, _)| r) != Some(state_reg) {
                        return Some(cur);
                    }
                } else {
                    return Some(cur);
                }
            }

            match &block.terminator {
                Terminator::Branch {
                    cond,
                    true_target,
                    false_target,
                } => {
                    if let Some(res) = Self::eval_condition_with_state(cond, state_reg, state_val) {
                        cur = if res { *true_target } else { *false_target };
                    } else {
                        return Some(cur);
                    }
                }
                Terminator::Goto(target) => {
                    cur = *target;
                }
                _ => return Some(cur),
            }
        }

        None
    }

    fn extract_state_comparison(cond: &IrExpr) -> Option<(u16, i64)> {
        match cond {
            IrExpr::Binary { op, left, right } => {
                if matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                ) {
                    match (&**left, &**right) {
                        (IrExpr::Register(r), IrExpr::Number(n)) => Some((*r, *n as i64)),
                        (IrExpr::Number(n), IrExpr::Register(r)) => Some((*r, *n as i64)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Rewires basic block jumps to target handlers and prunes the dispatcher tree.
    pub fn unflatten<'arena>(cfg: &mut ControlFlowGraph<'arena>, disp: &DispatcherInfo) {
        let state_reg = disp.state_reg;
        let header = disp.header;
        let state_map = &disp.state_map;

        let block_ids: Vec<BlockId> = cfg.blocks.keys().copied().collect();

        for bid in block_ids {
            if bid == header {
                continue;
            }

            let Some(block) = cfg.blocks.get_mut(&bid) else {
                continue;
            };

            // Check if this block ends by jumping to the dispatcher
            let jumps_to_disp = match block.terminator {
                Terminator::Goto(target) => target == header,
                _ => false,
            };

            if jumps_to_disp {
                // Find the state variable update in this block
                let mut state_assigned = None;
                let mut assign_idx = None;

                for (idx, inst) in block.instructions.iter().enumerate().rev() {
                    if let IrInst::Assign { dest, src } = inst {
                        if Some(*dest) == state_reg {
                            if let IrExpr::Number(n) = src {
                                state_assigned = Some(*n as i64);
                                assign_idx = Some(idx);
                                break;
                            }
                        }
                    }
                }

                if let (Some(state_val), Some(idx)) = (state_assigned, assign_idx) {
                    if let Some(&target_bid) = state_map.get(&state_val) {
                        // Strip the dead state assignment
                        block.instructions.remove(idx);
                        // Rewire jump directly to target handler
                        block.terminator = Terminator::Goto(target_bid);
                    }
                }
            }
        }

        // Retarget Entry Block if it initializes state and jumps to dispatcher
        if let Some(entry_block) = cfg.blocks.get_mut(&cfg.entry) {
            if entry_block.terminator == Terminator::Goto(header) {
                let mut init_state = None;
                let mut assign_idx = None;
                for (idx, inst) in entry_block.instructions.iter().enumerate().rev() {
                    if let IrInst::Assign { dest, src } = inst {
                        if Some(*dest) == state_reg {
                            if let IrExpr::Number(n) = src {
                                init_state = Some(*n as i64);
                                assign_idx = Some(idx);
                                break;
                            }
                        }
                    }
                }

                if let (Some(state_val), Some(idx)) = (init_state, assign_idx) {
                    if let Some(&target_bid) = state_map.get(&state_val) {
                        entry_block.instructions.remove(idx);
                        entry_block.terminator = Terminator::Goto(target_bid);
                    }
                }
            }
        }

        cfg.recompute_successors();
        prune_unreachable(cfg);
        thread_empty(cfg);
    }
}
