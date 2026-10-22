//! Variable naming, scope limiter (<= 180 locals), and Roblox semantic inference.

pub mod inference;
pub mod scope;

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashSet};

use crate::cfg::ir::{BlockId, ControlFlowGraph, IrInst};
use crate::ssa::{DefSite, SsaContext, VariableWeb};

pub use inference::{
    lookup_datatype, lookup_global, lookup_method, lookup_signal_param,
    singularize, to_camel_case, HintPriority, RobloxSemanticInference, SemanticHint,
    LUAU_KEYWORDS,
};
pub use scope::{
    declared_here, fit_locals, hoistable_names, limit_locals, names_anywhere,
    split_block, unlocalize, wrap_segments, LUAU_LOCAL_BUDGET, LUAU_MAX_LOCALS,
};

/// Live range interval [start_pc, end_pc].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct LiveInterval {
    pub start_pc: usize,
    pub end_pc: usize,
}

impl LiveInterval {
    #[inline]
    pub const fn new(start_pc: usize, end_pc: usize) -> Self {
        Self { start_pc, end_pc }
    }

    #[inline]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_pc <= other.end_pc && other.start_pc <= self.end_pc
    }

    #[inline]
    pub fn contains(&self, pc: usize) -> bool {
        pc >= self.start_pc && pc <= self.end_pc
    }

    #[inline]
    pub fn extend_to(&mut self, pc: usize) {
        if pc < self.start_pc {
            self.start_pc = pc;
        }
        if pc > self.end_pc {
            self.end_pc = pc;
        }
    }
}

/// Scope tree structure holding variable webs and body statements.
#[derive(Clone, Debug)]
pub struct ScopeTree<'arena> {
    pub active_webs: Vec<usize>,
    pub local_count: usize,
    pub body: Vec<crate::codegen::LuauStat<'arena>>,
}

/// Configuration parameters for the variable namer.
#[derive(Clone, Debug)]
pub struct NamerConfig {
    /// Prefix used for non-semantic recycled registers (default: "v").
    pub reg_prefix: String,
    /// Whether semantic inference is enabled (default: true).
    pub enable_semantic_inference: bool,
    /// Maximum local variable budget before triggering scope splitting (default: 180).
    pub max_local_budget: usize,
}

impl Default for NamerConfig {
    fn default() -> Self {
        Self {
            reg_prefix: "v".to_string(),
            enable_semantic_inference: true,
            max_local_budget: LUAU_LOCAL_BUDGET,
        }
    }
}

/// An active interval entry in the linear scan priority queue.
#[derive(Clone, Eq, PartialEq)]
struct ActiveEntry {
    end_pc: usize,
    web_idx: usize,
    name: String,
    color: Option<u32>,
}

impl Ord for ActiveEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-Heap ordered by end_pc ascending
        other.end_pc.cmp(&self.end_pc).then_with(|| self.web_idx.cmp(&other.web_idx))
    }
}

impl PartialOrd for ActiveEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Namer engine executing Linear Scan interval coloring.
pub struct IntervalNamer {
    config: NamerConfig,
}

impl IntervalNamer {
    pub fn new(config: NamerConfig) -> Self {
        Self { config }
    }

    /// Colors and assigns variable names to all webs in O(N log N) without O(N^2) matrices.
    pub fn assign_names(&self, webs: &mut [VariableWeb]) {
        if webs.is_empty() {
            return;
        }

        // 1. Identify upvalues and assign dedicated names
        let mut reserved_names: HashSet<String> = HashSet::new();
        for web in webs.iter_mut() {
            if web.is_upvalue {
                let name = if let Some(hint) = &web.semantic_hint {
                    hint.name.clone()
                } else {
                    format!("up_{}", web.original_reg)
                };
                web.assigned_name = name.clone();
                reserved_names.insert(name);
            }
        }

        // 2. Identify loop variables and assign isolated names
        for web in webs.iter_mut() {
            if web.is_loop_var && web.assigned_name.is_empty() {
                let name = if let Some(hint) = &web.semantic_hint {
                    hint.name.clone()
                } else {
                    format!("i_{}", web.original_reg)
                };
                web.assigned_name = name.clone();
                reserved_names.insert(name);
            }
        }

        // 3. Index remaining webs to sort
        let mut order: Vec<usize> = (0..webs.len())
            .filter(|&idx| webs[idx].assigned_name.is_empty())
            .collect();

        // Sort by start_pc, then end_pc, then index
        order.sort_unstable_by(|&a, &b| {
            webs[a]
                .live_interval
                .0
                .cmp(&webs[b].live_interval.0)
                .then_with(|| webs[a].live_interval.1.cmp(&webs[b].live_interval.1))
                .then_with(|| a.cmp(&b))
        });

        let mut active_heap: BinaryHeap<ActiveEntry> = BinaryHeap::new();
        let mut active_names: HashSet<String> = HashSet::new();
        let mut free_colors: BTreeSet<u32> = BTreeSet::new();
        let mut next_color: u32 = 0;

        // 4. Linear Scan Sweep
        for &web_idx in &order {
            let start_pc = webs[web_idx].live_interval.0;
            let end_pc = webs[web_idx].live_interval.1;

            // Expire old intervals
            while let Some(top) = active_heap.peek() {
                if top.end_pc < start_pc {
                    let entry = active_heap.pop().unwrap();
                    active_names.remove(&entry.name);
                    if let Some(col) = entry.color {
                        free_colors.insert(col);
                    }
                } else {
                    break;
                }
            }

            // Allocate variable name
            let (assigned_name, allocated_color) = if self.config.enable_semantic_inference
                && webs[web_idx]
                    .semantic_hint
                    .as_ref()
                    .map_or(false, |h| h.priority >= HintPriority::Weak)
            {
                let base = webs[web_idx].semantic_hint.as_ref().unwrap().name.clone();
                let mut cand = base.clone();
                let mut suffix = 1;
                while active_names.contains(&cand)
                    || reserved_names.contains(&cand)
                    || LUAU_KEYWORDS.contains(&cand.as_str())
                {
                    suffix += 1;
                    cand = format!("{}{}", base, suffix);
                }
                (cand, None)
            } else {
                // Non-semantic colored register
                let color = if let Some(&c) = free_colors.iter().next() {
                    free_colors.remove(&c);
                    c
                } else {
                    let c = next_color;
                    next_color += 1;
                    c
                };

                let mut cand = format!("{}{}", self.config.reg_prefix, color);
                let mut extra_color = color;
                while active_names.contains(&cand)
                    || reserved_names.contains(&cand)
                    || LUAU_KEYWORDS.contains(&cand.as_str())
                {
                    extra_color += 1;
                    cand = format!("{}{}", self.config.reg_prefix, extra_color);
                }
                (cand, Some(color))
            };

            webs[web_idx].assigned_name = assigned_name.clone();
            active_names.insert(assigned_name.clone());
            active_heap.push(ActiveEntry {
                end_pc,
                web_idx,
                name: assigned_name,
                color: allocated_color,
            });
        }
    }
}

/// Executes semantic hint inference across the CFG and runs linear scan naming.
pub fn run_namer<'arena>(ssa_ctx: &mut SsaContext, cfg: &ControlFlowGraph<'arena>) {
    // 1. Traverse CFG blocks and instructions to extract semantic hints
    let mut sorted_bids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
    sorted_bids.sort_by_key(|b| b.0);

    for bid in sorted_bids {
        let block = &cfg.blocks[&bid];
        let base_pc = block.pc_range.0;

        for (i, inst) in block.instructions.iter().enumerate() {
            let pc = base_pc + i;

            match inst {
                IrInst::Assign { dest, src } => {
                    if let Some(hint) = RobloxSemanticInference::infer_from_expr(src, pc, 0) {
                        let site = DefSite::inst(bid, i, *dest);
                        if let Some(&web_id) = ssa_ctx.def_to_web.get(&site) {
                            if let Some(web) = ssa_ctx.get_web_mut(web_id) {
                                web.semantic_hint = match web.semantic_hint.take() {
                                    Some(existing) => Some(existing.merge(hint)),
                                    None => Some(hint),
                                };
                            }
                        }
                    }
                }
                IrInst::CallStmt {
                    callee,
                    args,
                    ret_dest,
                    ret_count,
                } => {
                    if let Some(dest) = ret_dest {
                        let count = (*ret_count).max(1) as usize;
                        for ret_idx in 0..count {
                            let reg = *dest + ret_idx as u16;
                            if let Some(hint) =
                                RobloxSemanticInference::infer_from_call(callee, args, pc, ret_idx)
                            {
                                let site = DefSite::inst(bid, i, reg);
                                if let Some(&web_id) = ssa_ctx.def_to_web.get(&site) {
                                    if let Some(web) = ssa_ctx.get_web_mut(web_id) {
                                        web.semantic_hint = match web.semantic_hint.take() {
                                            Some(existing) => Some(existing.merge(hint)),
                                            None => Some(hint),
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // 2. Run interval namer coloring
    let namer = IntervalNamer::new(NamerConfig::default());
    namer.assign_names(&mut ssa_ctx.webs);
}
