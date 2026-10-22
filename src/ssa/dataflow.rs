//! Reaching Definitions Dataflow Analysis & Event Extraction.
//!
//! Computes reaching definition sets across basic blocks using Reverse Post-Order (RPO)
//! iteration and preserves open upvalues across closure boundaries.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::cfg::ir::{BasicBlock, BlockId, ControlFlowGraph, IrExpr, IrInst, Terminator};
use super::{DefSite, ProgramPoint};

/// Key distinguishing register definitions from active open upvalues.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DefKey {
    /// Reaching definitions for virtual register `reg`.
    Reg(u16),
    /// Reaching definitions captured into an actively open upvalue for `reg`.
    OpenUpvalue(u16),
}

pub type DefSet = BTreeSet<DefSite>;
pub type StateMap = HashMap<DefKey, DefSet>;

/// Event kind occurring at a discrete program point.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum EventKind {
    /// Capture by reference in a closure ("c").
    Capture,
    /// Read/use of a register ("u").
    Use,
    /// Write/definition of a register ("d").
    Def,
    /// Explicit close of active upvalues at and above register ("x").
    Close,
}

/// An extracted event for dataflow tracking.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockEvent {
    pub point: ProgramPoint,
    pub kind: EventKind,
    pub reg: u16,
}

/// Traverses an IrExpr recursively, collecting all referenced virtual registers.
pub fn collect_expr_uses<'arena>(expr: &IrExpr<'arena>, uses: &mut Vec<u16>) {
    match expr {
        IrExpr::Register(reg) => {
            uses.push(*reg);
        }
        IrExpr::Unary { expr, .. } => {
            collect_expr_uses(expr, uses);
        }
        IrExpr::Binary { left, right, .. } => {
            collect_expr_uses(left, uses);
            collect_expr_uses(right, uses);
        }
        IrExpr::TableGet { table, key } => {
            collect_expr_uses(table, uses);
            collect_expr_uses(key, uses);
        }
        IrExpr::Call { callee, args } => {
            collect_expr_uses(callee, uses);
            for arg in *args {
                collect_expr_uses(arg, uses);
            }
        }
        IrExpr::IfExp { cond, then_expr, else_expr } => {
            collect_expr_uses(cond, uses);
            collect_expr_uses(then_expr, uses);
            collect_expr_uses(else_expr, uses);
        }
        IrExpr::Closure { upvalues, .. } => {
            for upval in *upvalues {
                collect_expr_uses(upval, uses);
            }
        }
        IrExpr::Phi(entries) => {
            for (_, branch_expr) in *entries {
                collect_expr_uses(branch_expr, uses);
            }
        }
        IrExpr::Nil
        | IrExpr::Boolean(_)
        | IrExpr::Number(_)
        | IrExpr::String(_)
        | IrExpr::Upvalue(_)
        | IrExpr::Global(_)
        | IrExpr::Constant(_)
        | IrExpr::NewTable { .. }
        | IrExpr::Vararg => {}
    }
}

/// Traverses an IrExpr recursively, collecting all registers captured by reference.
pub fn collect_expr_captures<'arena>(expr: &IrExpr<'arena>, captures: &mut Vec<u16>) {
    match expr {
        IrExpr::Closure { upvalues, .. } => {
            for upval in *upvalues {
                if let IrExpr::Register(r) = upval {
                    captures.push(*r);
                } else {
                    collect_expr_captures(upval, captures);
                }
            }
        }
        IrExpr::Unary { expr, .. } => collect_expr_captures(expr, captures),
        IrExpr::Binary { left, right, .. } => {
            collect_expr_captures(left, captures);
            collect_expr_captures(right, captures);
        }
        IrExpr::TableGet { table, key } => {
            collect_expr_captures(table, captures);
            collect_expr_captures(key, captures);
        }
        IrExpr::Call { callee, args } => {
            collect_expr_captures(callee, captures);
            for arg in *args {
                collect_expr_captures(arg, captures);
            }
        }
        IrExpr::IfExp { cond, then_expr, else_expr } => {
            collect_expr_captures(cond, captures);
            collect_expr_captures(then_expr, captures);
            collect_expr_captures(else_expr, captures);
        }
        _ => {}
    }
}

/// Extracts block events following the invariant: "c" -> "u" -> "d" -> "x".
pub fn extract_block_events<'arena>(block: &BasicBlock<'arena>) -> Vec<BlockEvent> {
    let mut events = Vec::new();

    // 1. Scan statements
    for (i, inst) in block.instructions.iter().enumerate() {
        let point = ProgramPoint::Stmt(block.id, i);

        let mut caps = Vec::new();
        let mut uses = Vec::new();
        let mut defs = Vec::new();
        let mut closes = Vec::new();

        match inst {
            IrInst::Assign { dest, src } => {
                collect_expr_captures(src, &mut caps);
                collect_expr_uses(src, &mut uses);
                defs.push(*dest);
            }
            IrInst::TableSet { table, key, value } => {
                collect_expr_captures(table, &mut caps);
                collect_expr_captures(key, &mut caps);
                collect_expr_captures(value, &mut caps);

                collect_expr_uses(table, &mut uses);
                collect_expr_uses(key, &mut uses);
                collect_expr_uses(value, &mut uses);
            }
            IrInst::SetUpvalue { src, .. } | IrInst::SetGlobal { src, .. } => {
                collect_expr_captures(src, &mut caps);
                collect_expr_uses(src, &mut uses);
            }
            IrInst::CallStmt { callee, args, ret_dest, ret_count } => {
                collect_expr_captures(callee, &mut caps);
                collect_expr_uses(callee, &mut uses);
                for arg in *args {
                    collect_expr_captures(arg, &mut caps);
                    collect_expr_uses(arg, &mut uses);
                }
                if let Some(dest) = *ret_dest {
                    let count = (*ret_count).max(1) as u16;
                    for k in 0..count {
                        defs.push(dest + k);
                    }
                }
            }
            IrInst::SetList { table, values, .. } => {
                collect_expr_captures(table, &mut caps);
                collect_expr_uses(table, &mut uses);
                for v in *values {
                    collect_expr_captures(v, &mut caps);
                    collect_expr_uses(v, &mut uses);
                }
            }
            IrInst::CloseUpvalue(reg) => {
                closes.push(*reg);
            }
            IrInst::Comment(_) | IrInst::Nop => {}
        }

        // Strict ordering invariant: "c" -> "u" -> "d" -> "x"
        for r in caps {
            events.push(BlockEvent { point, kind: EventKind::Capture, reg: r });
        }
        for r in uses {
            events.push(BlockEvent { point, kind: EventKind::Use, reg: r });
        }
        for r in defs {
            events.push(BlockEvent { point, kind: EventKind::Def, reg: r });
        }
        for r in closes {
            events.push(BlockEvent { point, kind: EventKind::Close, reg: r });
        }
    }

    // 2. Scan terminator
    let term_point = ProgramPoint::Terminator(block.id);
    let mut term_caps = Vec::new();
    let mut term_uses = Vec::new();
    let mut term_defs = Vec::new();

    match &block.terminator {
        Terminator::Branch { cond, .. } => {
            collect_expr_captures(cond, &mut term_caps);
            collect_expr_uses(cond, &mut term_uses);
        }
        Terminator::Return(values) => {
            for v in *values {
                collect_expr_captures(v, &mut term_caps);
                collect_expr_uses(v, &mut term_uses);
            }
        }
        Terminator::ForLoop { var, start, limit, step, .. } => {
            collect_expr_captures(start, &mut term_caps);
            collect_expr_uses(start, &mut term_uses);

            collect_expr_captures(limit, &mut term_caps);
            collect_expr_uses(limit, &mut term_uses);

            if let Some(s) = step {
                collect_expr_captures(s, &mut term_caps);
                collect_expr_uses(s, &mut term_uses);
            }
            term_defs.push(*var);
        }
        Terminator::ForInLoop { vars, iterators, .. } => {
            for it in iterators {
                collect_expr_captures(it, &mut term_caps);
                collect_expr_uses(it, &mut term_uses);
            }
            term_defs.extend(vars.iter().copied());
        }
        Terminator::Goto(_) | Terminator::Crash | Terminator::Unreachable => {}
    }

    for r in term_caps {
        events.push(BlockEvent { point: term_point, kind: EventKind::Capture, reg: r });
    }
    for r in term_uses {
        events.push(BlockEvent { point: term_point, kind: EventKind::Use, reg: r });
    }
    for r in term_defs {
        events.push(BlockEvent { point: term_point, kind: EventKind::Def, reg: r });
    }

    events
}

/// Computes the Reverse Post-Order (RPO) block sequence of a CFG.
pub fn compute_cfg_rpo<'arena>(cfg: &ControlFlowGraph<'arena>) -> Vec<BlockId> {
    let mut visited = HashSet::new();
    let mut post_order = Vec::with_capacity(cfg.blocks.len());

    fn dfs<'arena>(
        bid: BlockId,
        cfg: &ControlFlowGraph<'arena>,
        visited: &mut HashSet<BlockId>,
        post_order: &mut Vec<BlockId>,
    ) {
        if !visited.insert(bid) {
            return;
        }
        if let Some(block) = cfg.blocks.get(&bid) {
            for &succ in &block.successors {
                dfs(succ, cfg, visited, post_order);
            }
        }
        post_order.push(bid);
    }

    dfs(cfg.entry, cfg, &mut visited, &mut post_order);
    post_order.reverse();
    post_order
}

/// Applies transfer function through a sequence of block events.
pub fn transfer_block(
    events: &[BlockEvent],
    in_state: &StateMap,
) -> StateMap {
    let mut cur = in_state.clone();

    for ev in events {
        match ev.kind {
            EventKind::Def => {
                let site = match ev.point {
                    ProgramPoint::Param => DefSite::param(ev.reg),
                    ProgramPoint::Stmt(bid, idx) => DefSite::inst(bid, idx, ev.reg),
                    ProgramPoint::Terminator(bid) => DefSite::term(bid, ev.reg),
                };
                let mut defs = BTreeSet::new();
                defs.insert(site);
                cur.insert(DefKey::Reg(ev.reg), defs);
            }
            EventKind::Capture => {
                let reg_key = DefKey::Reg(ev.reg);
                let open_key = DefKey::OpenUpvalue(ev.reg);

                if !cur.contains_key(&reg_key) || cur[&reg_key].is_empty() {
                    let syn = DefSite::synthetic_capture(ev.point, ev.reg);
                    let mut defs = BTreeSet::new();
                    defs.insert(syn);
                    cur.insert(reg_key, defs);
                }

                let active_defs = cur[&reg_key].clone();
                cur.entry(open_key).or_default().extend(active_defs);
            }
            EventKind::Close => {
                cur.remove(&DefKey::OpenUpvalue(ev.reg));
            }
            EventKind::Use => {}
        }
    }

    cur
}

/// Results of the Reaching Definitions analysis.
pub struct ReachingDefsResult {
    pub in_states: HashMap<BlockId, StateMap>,
    pub out_states: HashMap<BlockId, StateMap>,
    pub events: HashMap<BlockId, Vec<BlockEvent>>,
}

/// Solves reaching definitions dataflow across all blocks in the CFG.
pub fn run_reaching_definitions<'arena>(
    cfg: &ControlFlowGraph<'arena>,
    num_params: u8,
) -> ReachingDefsResult {
    let mut events = HashMap::with_capacity(cfg.blocks.len());
    for (&bid, block) in &cfg.blocks {
        events.insert(bid, extract_block_events(block));
    }

    let order = compute_cfg_rpo(cfg);

    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::with_capacity(cfg.blocks.len());
    for &bid in cfg.blocks.keys() {
        preds.insert(bid, Vec::new());
    }
    for (&bid, block) in &cfg.blocks {
        for &succ in &block.successors {
            if let Some(list) = preds.get_mut(&succ) {
                list.push(bid);
            }
        }
    }

    let mut in_states: HashMap<BlockId, StateMap> = HashMap::with_capacity(cfg.blocks.len());
    let mut out_states: HashMap<BlockId, StateMap> = HashMap::with_capacity(cfg.blocks.len());
    for &bid in cfg.blocks.keys() {
        in_states.insert(bid, HashMap::new());
        out_states.insert(bid, HashMap::new());
    }

    if num_params > 0 {
        if let Some(entry_in) = in_states.get_mut(&cfg.entry) {
            for p in 0..num_params as u16 {
                let mut s = BTreeSet::new();
                s.insert(DefSite::param(p));
                entry_in.insert(DefKey::Reg(p), s);
            }
        }
    }

    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 100 {
        changed = false;
        iterations += 1;

        for &bid in &order {
            let mut inn: StateMap = HashMap::new();
            if let Some(p_list) = preds.get(&bid) {
                for &p in p_list {
                    if let Some(p_out) = out_states.get(&p) {
                        for (&k, def_set) in p_out {
                            inn.entry(k).or_default().extend(def_set.iter().copied());
                        }
                    }
                }
            }

            if bid == cfg.entry && num_params > 0 {
                for p in 0..num_params as u16 {
                    inn.entry(DefKey::Reg(p))
                        .or_default()
                        .insert(DefSite::param(p));
                }
            }

            in_states.insert(bid, inn.clone());

            let out = transfer_block(&events[&bid], &inn);
            if out_states.get(&bid) != Some(&out) {
                out_states.insert(bid, out);
                changed = true;
            }
        }
    }

    ReachingDefsResult {
        in_states,
        out_states,
        events,
    }
}

/// Traverses an IrExpr recursively, collecting all referenced upvalue indices.
pub fn collect_expr_upvalues<'arena>(expr: &IrExpr<'arena>, upvalues: &mut BTreeSet<u16>) {
    match expr {
        IrExpr::Upvalue(u) => {
            upvalues.insert(*u);
        }
        IrExpr::Unary { expr, .. } => collect_expr_upvalues(expr, upvalues),
        IrExpr::Binary { left, right, .. } => {
            collect_expr_upvalues(left, upvalues);
            collect_expr_upvalues(right, upvalues);
        }
        IrExpr::TableGet { table, key } => {
            collect_expr_upvalues(table, upvalues);
            collect_expr_upvalues(key, upvalues);
        }
        IrExpr::Call { callee, args } => {
            collect_expr_upvalues(callee, upvalues);
            for arg in *args {
                collect_expr_upvalues(arg, upvalues);
            }
        }
        IrExpr::IfExp { cond, then_expr, else_expr } => {
            collect_expr_upvalues(cond, upvalues);
            collect_expr_upvalues(then_expr, upvalues);
            collect_expr_upvalues(else_expr, upvalues);
        }
        IrExpr::Closure { upvalues: closure_upvals, .. } => {
            for upval in *closure_upvals {
                collect_expr_upvalues(upval, upvalues);
            }
        }
        IrExpr::Phi(entries) => {
            for (_, branch_expr) in *entries {
                collect_expr_upvalues(branch_expr, upvalues);
            }
        }
        _ => {}
    }
}

/// Collects all upvalue indices that are genuinely read or written across the CFG.
pub fn collect_referenced_upvalues<'arena>(cfg: &ControlFlowGraph<'arena>) -> BTreeSet<u16> {
    let mut referenced = BTreeSet::new();

    for block in cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                IrInst::Assign { src, .. } => {
                    collect_expr_upvalues(src, &mut referenced);
                }
                IrInst::TableSet { table, key, value } => {
                    collect_expr_upvalues(table, &mut referenced);
                    collect_expr_upvalues(key, &mut referenced);
                    collect_expr_upvalues(value, &mut referenced);
                }
                IrInst::SetUpvalue { upvalue_idx, src } => {
                    referenced.insert(*upvalue_idx);
                    collect_expr_upvalues(src, &mut referenced);
                }
                IrInst::SetGlobal { src, .. } => {
                    collect_expr_upvalues(src, &mut referenced);
                }
                IrInst::CallStmt { callee, args, .. } => {
                    collect_expr_upvalues(callee, &mut referenced);
                    for arg in *args {
                        collect_expr_upvalues(arg, &mut referenced);
                    }
                }
                IrInst::SetList { table, values, .. } => {
                    collect_expr_upvalues(table, &mut referenced);
                    for val in *values {
                        collect_expr_upvalues(val, &mut referenced);
                    }
                }
                _ => {}
            }
        }

        match &block.terminator {
            Terminator::Branch { cond, .. } => {
                collect_expr_upvalues(cond, &mut referenced);
            }
            Terminator::Return(vals) => {
                for val in *vals {
                    collect_expr_upvalues(val, &mut referenced);
                }
            }
            Terminator::ForLoop { start, limit, step, .. } => {
                collect_expr_upvalues(start, &mut referenced);
                collect_expr_upvalues(limit, &mut referenced);
                if let Some(s) = step {
                    collect_expr_upvalues(s, &mut referenced);
                }
            }
            Terminator::ForInLoop { iterators, .. } => {
                for iter_expr in iterators {
                    collect_expr_upvalues(iter_expr, &mut referenced);
                }
            }
            _ => {}
        }
    }

    referenced
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;

    #[test]
    fn test_collect_referenced_upvalues() {
        let bump = Bump::new();
        let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

        let mut b0 = BasicBlock::new(
            BlockId(0),
            (0, 2),
            Terminator::Return(bump.alloc_slice_clone(&[IrExpr::Upvalue(7)])),
        );

        b0.instructions.push(IrInst::Assign {
            dest: 0,
            src: IrExpr::Upvalue(3),
        });

        b0.instructions.push(IrInst::SetUpvalue {
            upvalue_idx: 12,
            src: IrExpr::Register(0),
        });

        cfg.add_block(b0);
        let refs = collect_referenced_upvalues(&cfg);
        assert_eq!(refs.len(), 3);
        assert!(refs.contains(&3));
        assert!(refs.contains(&7));
        assert!(refs.contains(&12));
        assert!(!refs.contains(&0));
    }
}


